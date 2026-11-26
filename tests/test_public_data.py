"""Loaders and analysis of the public UCI datasets, on tiny files in each dataset's own format."""
import tempfile
import unittest
from pathlib import Path

from smart_house.analytics.public_data import DatasetWeather, analyze, load_appliances, load_occupancy, load_sml2010

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/models/rollout-v2.pt'


def write_occupancy(directory, minutes=60):
    header = '"date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"\n'
    for name, start in (('datatest.txt', 0), ('datatraining.txt', minutes), ('datatest2.txt', 2 * minutes)):
        lines = [header]
        for m in range(start, start + minutes):
            occupied = int(m % 120 < 60)
            lines.append(f'"{m}","2015-02-04 {8 + m // 60:02d}:{m % 60:02d}:00",21.5,30.0,300,{500 + 4 * (m % 120 if occupied else 120 - m % 120)},0.004,{occupied}\n')
        (Path(directory) / name).write_text(''.join(lines))
    # One duplicate timestamp across files must be merged.
    with open(Path(directory) / 'datatest2.txt', 'a') as f:
        f.write('"999","2015-02-04 08:00:00",21.5,30.0,300,500,0.004,1\n')


def write_appliances(directory, rows=12):
    cols = ['date', 'Appliances', 'lights'] + [f'{k}{i}' for i in range(1, 10) for k in ('T', 'RH_')] + \
        ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
    lines = [','.join(f'"{c}"' for c in cols)]
    for i in range(rows):
        values = {'date': f'2016-01-11 {17 + i // 6:02d}:{(i % 6) * 10:02d}:00', 'Appliances': '60', 'lights': '0',
                  'T_out': '5.0', 'Press_mm_hg': '733', 'RH_out': '90', 'Windspeed': '2', 'Visibility': '60',
                  'Tdewpoint': '3', 'rv1': '1', 'rv2': '1'}
        for r in range(1, 10):
            values[f'T{r}'] = '30.0' if r == 6 else f'{20 + r * 0.1:.2f}'   # T6 is outdoors: must be excluded
            values[f'RH_{r}'] = '90' if r == 6 else '45'
        lines.append(','.join(f'"{values[c]}"' for c in cols))
    (Path(directory) / 'energydata_complete.csv').write_text('\n'.join(lines) + '\n')


def write_sml(directory, rows=8):
    head = '# 1:Date 2:Time ... 24:Day_Of_Week\n'
    body = ''.join(f'13/03/2012 {11 + i // 4:02d}:{(i % 4) * 15:02d} 19.0 18.5 0 420 430 40.0 41.0 80 100 0 600 1.4 '
                   f'9000 12000 95000 750 {1 if i == 3 else 0} 0 0 12.0 60.0 2\n' for i in range(rows))
    (Path(directory) / 'NEW-DATA-1.T15.txt').write_text(head + body)
    (Path(directory) / 'NEW-DATA-2.T15.txt').write_text(head)


class PublicDataTests(unittest.TestCase):
    def test_loaders_map_each_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_occupancy(tmp)
            rows = load_occupancy(tmp)
            self.assertEqual(len(rows), 180)
            self.assertEqual(rows[0]['timestamp'].utcoffset().total_seconds(), 3600)
            self.assertIn(rows[0]['total_occupants'], (0, 1))
            write_appliances(tmp)
            house = load_appliances(tmp)[0]
            self.assertAlmostEqual(house['temperature_c'], 20 + 0.1 * (1 + 2 + 3 + 4 + 5 + 7 + 8 + 9) / 8)
            self.assertEqual((house['humidity_pct'], house['outdoor_temperature_c'], house['wind_kmh']), (45.0, 5.0, 7.2))
            write_sml(tmp)
            sml = load_sml2010(tmp)
            self.assertEqual((len(sml), sml[0]['co2_ppm'], sml[0]['outdoor_temperature_c']), (8, 420.0, 12.0))
            self.assertTrue(sml[3]['ventilation_on'])
            weather, source = DatasetWeather(sml).get(sml[0]['timestamp'])
            self.assertEqual((source, weather.temperature_c), ('measured', 12.0))

    def test_analysis_report_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_sml(tmp, rows=40)
            report = analyze('sml2010', tmp, MODEL)
        self.assertEqual(report['summary']['records'], 40)
        self.assertEqual(report['summary']['rejected'], 0)
        self.assertEqual(report['shadow']['weather_sources'], {'measured': 40, 'fixed': 0})
        self.assertEqual(report['shadow']['network']['decisions'], 40)
        self.assertIn('params', report['calibration'])
        self.assertIn('doi', report['source'])


if __name__ == '__main__':
    unittest.main()
