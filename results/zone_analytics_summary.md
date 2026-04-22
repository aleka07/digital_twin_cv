# Zone Analytics Summary

- zones: `article_pack\zones\cam10_zones_final.json`
- clips: `clip_A, clip_B`
- membership rule: bbox center inside polygon

## clip_A
- `left_transit_lane`: entries=0, exits=0, avg_occ=0.0, max_occ=0, total_dwell_s=0
- `rack_buffer_area`: entries=2, exits=0, avg_occ=1.389, max_occ=2, total_dwell_s=41.667
- `machine_service_area`: entries=2, exits=0, avg_occ=1.022, max_occ=2, total_dwell_s=30.667
- `front_vat_staging`: entries=7, exits=5, avg_occ=2.344, max_occ=4, total_dwell_s=70.333
- `right_loading_area`: entries=4, exits=3, avg_occ=0.983, max_occ=2, total_dwell_s=29.5
- top dwell tracks:
  - zone=`rack_buffer_area` track=`8` class=`person` dwell_s=30.167
  - zone=`front_vat_staging` track=`2` class=`vat` dwell_s=30.0
  - zone=`front_vat_staging` track=`3` class=`vat` dwell_s=30.0
  - zone=`machine_service_area` track=`1` class=`vat` dwell_s=30.0
  - zone=`right_loading_area` track=`6` class=`person` dwell_s=17.833

## clip_B
- `left_transit_lane`: entries=0, exits=0, avg_occ=0.0, max_occ=0, total_dwell_s=0
- `rack_buffer_area`: entries=0, exits=0, avg_occ=0.0, max_occ=0, total_dwell_s=0
- `machine_service_area`: entries=3, exits=3, avg_occ=0.206, max_occ=1, total_dwell_s=6.166
- `front_vat_staging`: entries=5, exits=2, avg_occ=3.106, max_occ=4, total_dwell_s=93.167
- `right_loading_area`: entries=3, exits=1, avg_occ=1.678, max_occ=2, total_dwell_s=50.333
- top dwell tracks:
  - zone=`front_vat_staging` track=`1` class=`vat` dwell_s=30.0
  - zone=`front_vat_staging` track=`3` class=`vat` dwell_s=30.0
  - zone=`right_loading_area` track=`5` class=`person` dwell_s=30.0
  - zone=`front_vat_staging` track=`2` class=`person` dwell_s=27.5
  - zone=`right_loading_area` track=`4` class=`person` dwell_s=20.333
