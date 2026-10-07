# G1 data-use draft

Retrieved 2026-10-07. This records source text and unresolved permissions; it does not grant access. No train-enquiry endpoint was requested. The live collector remains disabled.

| Source | Automated collection | Model training | Kaggle export | Portfolio display |
|---|---|---|---|---|
| NTES | Prohibited systematic database downloads without prior written permission | Not established; requires permission | Not permitted by the quoted copying restrictions without permission | Requires permission and acknowledgement |
| etrain / Tripozo | Unclear; no explicit automated-access grant found | Unclear | Restricted transmission of copied content; permission needed | Restricted copied-content use; permission needed |
| Pinned GTFS repository | Unclear: no repository licence found | Unclear | Unclear | Unclear |
| OSM / Geofabrik extract | ODbL permits data use; use bulk extracts, comply with host download policy | ODbL conditions apply to derived databases; prediction models do not use OSM in v1 | ODbL attribution/share-alike conditions apply | Permitted with OSM attribution and licence notice |
| Protomaps basemap | Published download/cutout workflow; self-host rather than hotlink | Not a prediction input in v1 | Produced-work attribution applies; no need to export tiles to Kaggle | Permitted with OpenStreetMap attribution |
| datameet railway data | CC0 | CC0 | CC0 | CC0; credit encouraged |

## NTES

URL: https://enquiry.indianrail.gov.in/mntes/disclaimerDisplay.html (the Disclaimer link on the landing page). HTTP 200.

> You may not (whether directly or through the use of any software program) create a database in electronic or structured manual form by regularly or systematically downloading and storing all or any part of the pages from this site.

> No part of The Site may be reproduced or transmitted to or stored in any other web site [...] nor included in any public or private electronic retrieval system or service without Government of India, Ministry of Railway's prior written permission.

> Material featured on this Website may be reproduced free of charge after taking proper permission by sending a mail to us.

The disclaimer also disclaims timestamp accuracy, completeness and timeliness. It supplies no hard publication-lag guarantee.

## etrain / Tripozo

etrain links to Tripozo for account and booking services. The Terms and Conditions link on https://www.tripozo.com/ resolves to https://www.tripozo.com/page/termsconditions (HTTP 200).

> User shall not distribute, exchange, modify, sell or transmit any content or material you copy from this Site. User cannot use this Content for any other purpose, except as specified herein.

> Tripozo grants you a non-exclusive, non-transferable, limited right to enter, view and use and transact on this Site.

Applicability to etrain train-history data and automated collection is **unclear**. The earlier spec's “Personal Non-Commercial Use” statement was not reproduced in the current retrieved terms; it is not treated as a grant. The attempted etrain `/page/terms`, `/terms` and `/page/termsconditions` URLs returned 404.

## GTFS

Repository: https://github.com/Neo2308/indianrailways-gtfs/tree/b340b7048e99c18330ad333e8210d61161ac7a43

The GitHub licence endpoint for that revision returned 404. Its README says:

> (Please open an issue/PR to add your project here using this data)

This invitation is not an explicit data licence. The upstream data-use rights remain **unclear**; the local import is not permission to redistribute.

## OSM

URL: https://www.openstreetmap.org/copyright (HTTP 200).

> OpenStreetMap is open data, licensed under the Open Data Commons Open Database License (ODbL).

> You are free to copy, distribute, transmit and adapt our data, as long as you credit OpenStreetMap and its contributors. If you alter or build upon our data, you may distribute the result only under the same license.

Full licence: https://opendatacommons.org/licenses/odbl/
Map attribution: https://osmfoundation.org/wiki/Licence/Attribution_Guidelines

## Protomaps

URL: https://docs.protomaps.com/basemaps/downloads (HTTP 200).

> It's available as a single PMTiles archive, distributed as an Open Database License Produced Work (OpenStreetMap attribution required).

> Hotlinking to these downloads are discouraged. Instead, you should copy the tileset to your own Cloud Storage.

The documented partial-download workflow fits a self-hosted India cutout. Basemap attribution remains required in the dashboard.

## datameet

URL: https://raw.githubusercontent.com/datameet/railways/e0c538a1e41ae5eace454d2818902e1065608e16/README.md (HTTP 200).

> License
>
> [CC0](https://wiki.creativecommons.org/wiki/CC0)

This covers the repository's stations, trains and schedules data, not third-party live sources.

## CP-1 decision

The user chose **“Change to a permitted live source”** after reviewing the NTES restriction. NTES protocol discovery, fixture capture and the seven-day pilot are not approved. etrain backfill and Kaggle export remain unapproved. Research a provider with an explicit API/data-use grant before replacing the source-specific client/parser contracts. Offline implementation may continue.

The user subsequently chose a **RailRadar sandbox integration, offline only**: client/parser against documented shapes with MockTransport fixtures, behind `MSYND_LIVE`; no account creation or API requests. The monitor revises Tasks 7–11 and 20–25. [Integration details](railradar-sandbox.md).

RailRadar's https://railradar.in/docs explicitly advertises developer REST API integration with a free 1,000-request/month sandbox and Bearer authentication. https://railradar.in/terms describes an independent crowd-powered service and GPS telemetry grants from contributors to RailRadar. These terms do not explicitly grant API consumers historical retention, model training or Kaggle export; those uses remain unclear. The API's `/trains/{number}/live` documentation specifies `date` as the journey start date in IST. It offers `authoritative=true` to bypass caches; the prepared client uses `false`. No API request was made to verify runtime response quality or coverage.
