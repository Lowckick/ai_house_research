"""Real indoor sensor recordings from public UCI datasets, run through the same pipeline as live sensors.

Three CC BY 4.0 datasets from the UCI Machine Learning Repository:

* ``occupancy``  — Candanedo & Feldheim (2016), Energy and Buildings 112: an office in Mons, Belgium,
  one-minute temperature, humidity, light, CO₂ and photo-verified occupancy (Feb 2015).
* ``appliances`` — Candanedo, Feldheim & Deramaix (2017), Energy and Buildings 140: a low-energy house in
  Belgium, 10-minute temperature and humidity in nine rooms, outdoor weather from Chièvres airport (2016).
* ``sml2010``   — Romeu-Guallart & Zamora-Martinez (2014): a domotic house in Valencia, Spain, 15-minute
  temperature, CO₂ and humidity in two rooms and an outdoor sensor (Mar–May 2012).

Each loader returns rows with indoor values and, where measured, outdoor temperature and humidity.
``analyze`` validates the readings, replays them through the shadow-mode live channel with the measured
outdoor weather, summarises them, and calibrates the simulator. Device actions are not part of the
datasets, so the analysis covers conditions and recommendations, not the effect of following them.
"""
import csv
import datetime as dt
import json
import statistics
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

from smart_house.analytics.calibration import calibrate
from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService
from smart_house.analytics.live import LiveService
from smart_house.environment.readings import inspect_readings
from smart_house.environment.weather import WeatherSnapshot

BRUSSELS = ZoneInfo('Europe/Brussels')
SOURCES = {
    'occupancy': {'title': 'UCI Occupancy Detection', 'doi': '10.24432/C5X01N',
                  'url': 'https://archive.ics.uci.edu/dataset/357/occupancy+detection',
                  'citation': 'Candanedo L. M., Feldheim V. Accurate occupancy detection of an office room from light, '
                              'temperature, humidity and CO2 measurements using statistical learning models. '
                              'Energy and Buildings. 2016. Vol. 112. P. 28–39.'},
    'appliances': {'title': 'UCI Appliances Energy Prediction', 'doi': '10.24432/C5VC8G',
                   'url': 'https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction',
                   'citation': 'Candanedo L. M., Feldheim V., Deramaix D. Data driven prediction models of energy use '
                               'of appliances in a low-energy house. Energy and Buildings. 2017. Vol. 140. P. 81–97.'},
    'sml2010': {'title': 'UCI SML2010', 'doi': '10.24432/C5RS3S',
                'url': 'https://archive.ics.uci.edu/dataset/274/sml2010',
                'citation': 'Romeu-Guallart P., Zamora-Martinez F. SML2010 [Dataset]. UCI Machine Learning Repository. '
                            '2014.'},
}
INDOOR_ROOMS = ('1', '2', '3', '4', '5', '7', '8', '9')   # appliances: T6/RH_6 is outside the north wall


def load_occupancy(directory):
    """All three files, sorted and de-duplicated; occupancy is presence (0/1), not a head count."""
    rows = {}
    for name in ('datatest.txt', 'datatraining.txt', 'datatest2.txt'):
        with open(Path(directory) / name, newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream)
            next(reader)   # the header names 7 columns; each row starts with an unnamed row number
            for _, date, temperature, humidity, _light, co2, _ratio, occupancy in reader:
                stamp = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=BRUSSELS)
                rows[stamp] = {'timestamp': stamp, 'temperature_c': float(temperature),
                               'humidity_pct': float(humidity), 'co2_ppm': float(co2),
                               'total_occupants': int(occupancy)}
    return [rows[k] for k in sorted(rows)]


def load_appliances(directory):
    """House average of the eight indoor rooms with the measured outdoor weather."""
    rows = []
    with open(Path(directory) / 'energydata_complete.csv', newline='', encoding='utf-8') as stream:
        for record in csv.DictReader(stream):
            stamp = dt.datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=BRUSSELS)
            def value(key, record=record):
                return float(record[key].strip())
            rows.append({'timestamp': stamp,
                         'temperature_c': statistics.fmean(value('T' + r) for r in INDOOR_ROOMS),
                         'humidity_pct': statistics.fmean(value('RH_' + r) for r in INDOOR_ROOMS),
                         'outdoor_temperature_c': value('T_out'), 'outdoor_humidity_pct': value('RH_out'),
                         'wind_kmh': value('Windspeed') * 3.6})
    return rows


def load_sml2010(directory):
    """Dining-room sensors with the house's own outdoor sensor; timestamps are UTC."""
    rows = {}
    for name in ('NEW-DATA-1.T15.txt', 'NEW-DATA-2.T15.txt'):
        for line in (Path(directory) / name).read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            cells = line.split()
            stamp = dt.datetime.strptime(f'{cells[0]} {cells[1]}', '%d/%m/%Y %H:%M').replace(tzinfo=dt.timezone.utc)
            rows[stamp] = {'timestamp': stamp, 'temperature_c': float(cells[2]), 'co2_ppm': float(cells[5]),
                           'humidity_pct': float(cells[7]), 'outdoor_temperature_c': float(cells[21]),
                           'outdoor_humidity_pct': float(cells[22]),
                           'ventilation_on': any(float(c) > 0 for c in cells[18:21])}
    return [rows[k] for k in sorted(rows)]


LOADERS = {'occupancy': load_occupancy, 'appliances': load_appliances, 'sml2010': load_sml2010}


class DatasetWeather:
    """Weather source for the live channel: the dataset's own outdoor values at each timestamp."""

    def __init__(self, rows):
        self._by_time = {r['timestamp']: r for r in rows if 'outdoor_temperature_c' in r}

    def prefetch(self, timestamps):
        return 0

    def get(self, timestamp):
        row = self._by_time.get(timestamp)
        if row is None:
            from smart_house.environment.weather_archive import fixed_weather
            return fixed_weather(timestamp), 'fixed'
        local = timestamp.astimezone(BRUSSELS)
        return WeatherSnapshot(timestamp, row['outdoor_temperature_c'], row['outdoor_humidity_pct'],
                               row.get('wind_kmh', 0.0), 50.0, 0.0, 0.0, 7 <= local.hour < 18, 'cloudy'), 'measured'


def reading(row):
    """A row as a reading of the shared contract (values rounded as sensors report them)."""
    out = {'timestamp': row['timestamp'].isoformat(), 'temperature_c': round(row['temperature_c'], 2),
           'humidity_pct': round(min(100.0, max(0.0, row['humidity_pct'])), 2)}
    if 'co2_ppm' in row:
        out['co2_ppm'] = round(row['co2_ppm'], 1)
    if 'total_occupants' in row:
        out['total_occupants'] = row['total_occupants']
    return out


def _stats(rows, key):
    values = [r[key] for r in rows if key in r]
    if not values:
        return None
    return {'min': round(min(values), 2), 'median': round(statistics.median(values), 2), 'max': round(max(values), 2)}


def network_effect(payloads, readings):
    """How often the network changed the strengthened controller's decision, and the energy of each."""
    from smart_house.environment.indoor_state import DeviceState, IndoorState, device_power_kw
    from smart_house.ml.controller import tuned_deterministic_devices
    fields = ('hvac_mode', 'ventilation', 'air_purifier', 'humidifier_mode', 'cleaner_bot')
    changed, by_field, energy, tuned_energy = 0, {f: 0 for f in fields}, 0.0, 0.0
    times = [dt.datetime.fromisoformat(r['timestamp']) for r in payloads]
    for i, record in enumerate(payloads):
        indoor = IndoorState(**{**record['indoor'], 'timestamp': times[i]})
        weather = WeatherSnapshot(**{**record['weather'], 'timestamp': times[i]})
        base = tuned_deterministic_devices(indoor, weather)
        chosen = DeviceState(**{k: v for k, v in record['recommended_devices'].items()})
        diff = [f for f in fields if getattr(base, f) != getattr(chosen, f)]
        changed += bool(diff)
        for f in diff:
            by_field[f] += 1
        hours = min(30.0, (times[i + 1] - times[i]).total_seconds() / 60) / 60 if i + 1 < len(times) else 0.0
        energy += device_power_kw(chosen) * hours
        tuned_energy += device_power_kw(base) * hours
    days = max(1e-9, (times[-1] - times[0]).total_seconds() / 86400)
    return {'decisions': len(payloads), 'changed_tuned_decision': changed, 'changed_share': round(changed / len(payloads), 4),
            'changed_by_device': by_field, 'energy_kwh_day_network': round(energy / days, 2),
            'energy_kwh_day_tuned_controller': round(tuned_energy / days, 2)}


def analyze(name, directory, model, workdir=None):
    """Validation, data summary, shadow-mode replay and calibration of one dataset; JSON-safe report."""
    rows = LOADERS[name](directory)
    readings = [reading(r) for r in rows]
    _, validation = inspect_readings(json.dumps(readings[:5000]), '.json')  # validator caps one file at 5000
    rejected = 0
    for start in range(5000, len(readings), 5000):
        _, part = inspect_readings(json.dumps(readings[start:start + 5000]), '.json')
        rejected += len({e['record'] for e in part['errors']})
    rejected += len({e['record'] for e in validation['errors']})
    steps = [(b['timestamp'] - a['timestamp']).total_seconds() / 60 for a, b in zip(rows, rows[1:])]
    summary = {'records': len(rows), 'rejected': rejected,
               'first': rows[0]['timestamp'].isoformat(), 'last': rows[-1]['timestamp'].isoformat(),
               'days': round((rows[-1]['timestamp'] - rows[0]['timestamp']).total_seconds() / 86400, 1),
               'median_interval_min': statistics.median(steps),
               'temperature_c': _stats(rows, 'temperature_c'), 'humidity_pct': _stats(rows, 'humidity_pct'),
               'co2_ppm': _stats(rows, 'co2_ppm'), 'outdoor_temperature_c': _stats(rows, 'outdoor_temperature_c'),
               'occupied_share': (round(statistics.fmean(r['total_occupants'] for r in rows), 3)
                                  if 'total_occupants' in rows[0] else None),
               'ventilation_on_share': (round(statistics.fmean(r['ventilation_on'] for r in rows), 3)
                                        if 'ventilation_on' in rows[0] else None)}
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(workdir or tmp) / f'{name}-shadow.db'
        db.unlink(missing_ok=True)
        imports = ImportService(model, History(db), DatasetWeather(rows))
        clock = [rows[0]['timestamp']]
        live = LiveService(imports, clock=lambda: clock[0])
        for item in readings:
            clock[0] = dt.datetime.fromisoformat(item['timestamp'])
            live.ingest(name, item)
        hours = int((rows[-1]['timestamp'] - rows[0]['timestamp']).total_seconds() // 3600) + 1
        shadow = live.report(name, hours=hours)
        shadow.pop('alerts', None)   # the report keeps only the last 50; count every onset below
        with imports.history.connect() as conn:
            payloads = [json.loads(p[0]) for p in conn.execute('SELECT payload FROM live_readings WHERE sensor_id=?', (name,))]
        shadow['alert_onsets'] = {}
        for record in payloads:
            for a in record['new_alerts']:
                key = f"{a['metric']}:{a['level']}"
                shadow['alert_onsets'][key] = shadow['alert_onsets'].get(key, 0) + 1
        shadow['weather_sources'] = {s: sum(r['weather_source'] == s for r in payloads) for s in ('measured', 'fixed')}
        shadow['network'] = network_effect(payloads, readings)
    calibration_rows = [{'time': r['timestamp'], 'temperature_c': r['temperature_c'],
                         'humidity_pct': r['humidity_pct'], 'co2_ppm': r.get('co2_ppm', 450.0),
                         'total_occupants': float(r.get('total_occupants', 0)),
                         'outdoor_temperature_c': r.get('outdoor_temperature_c', 15.0),
                         'outdoor_humidity_pct': r.get('outdoor_humidity_pct', 60.0),
                         'outdoor_source': 'measured' if 'outdoor_temperature_c' in r else 'fixed'} for r in rows]
    calibration = calibrate(calibration_rows)
    return {'dataset': name, 'source': SOURCES[name], 'summary': summary, 'shadow': shadow,
            'calibration': calibration,
            'note': 'Public recordings without device actions: conditions and recommendations are analysed; '
                    'the effect of following the recommendations is not.'}
