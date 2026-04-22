# Zone Design Notes

## Recommended CAM10 Zone Set

These suggestions are based on visual review of `clip_A_000001.jpg` and `clip_B_000001.jpg`.

Recommended zones:

- `left_transit_lane`
  - Meaning: left-side movement corridor
  - Best for: person transit, crossing frequency
  - Keep it narrow so it does not eat the rack area

- `rack_buffer_area`
  - Meaning: upper-left staging / rack interaction area
  - Best for: temporary queueing, worker interaction with trays/racks
  - Keep it on the floor footprint, not on the wall or doorway frame

- `machine_service_area`
  - Meaning: main working floor around the central machine
  - Best for: service presence, machine-side dwell, main operation zone
  - This is the most important zone for the article

- `front_vat_staging`
  - Meaning: foreground vat parking / handling area
  - Best for: vat presence, vat dwell, worker-vat interaction around the front lane
  - Make it broad enough because vats occupy a large footprint

- `right_loading_area`
  - Meaning: right-side loading / ingredient / bag handling area
  - Best for: loading activity, ingredient-side presence, right-edge worker activity
  - Avoid overlapping too much with `machine_service_area`

## Simpler Alternative

If you want a cleaner and less ambiguous article setup, use only:

- `machine_service_area`
- `front_vat_staging`
- `right_loading_area`

This 3-zone version is easier to explain and usually stronger for a paper than too many micro-zones.

## Practical Annotation Advice

- Prefer floor/action polygons, not wall coverage.
- Keep overlaps minimal.
- Because current analytics uses bbox center, zones should not be too thin.
- For people, a future improvement would be to use bottom-center instead of bbox center.
- For vats, center-based membership is usually acceptable.

## Suggested Start File

- `article_pack/zones/cam10_zone_suggestions.json`

Use it as a draft and edit it in the interactive zone tool.
