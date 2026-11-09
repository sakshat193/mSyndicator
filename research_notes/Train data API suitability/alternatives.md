# Alternative Indian railway API suitability

## Which documented developer providers suit this tracker?

### Takeaway
RailRadar is the strongest documented fit found, especially if its aggregate map feed has usable live coverage. RailKit and Qrail explicitly offer developer APIs but their public plans are far below continuous fleet polling; none has been verified with an authenticated live response in this research.

### Cited Findings
- RailRadar documents account registration, API key generation, Bearer or X-API-Key authentication, and a 1,000 requests/month free sandbox. This is a documented developer access path, unlike a discovered consumer endpoint. [Official docs](https://railradar.in/docs)
- RailRadar documents `GET https://api.railradar.in/v1/trains/{number}/live`, optional journey date, optional cache bypass, location segment progress, speed, bearing, and actual versus scheduled station timestamps. Its example includes `lastUpdatedAt` and `currentLocation.isActualPosition`; these are documentation examples, not observed response values. [Live endpoint](https://railradar.in/docs/live-train-status)
- RailRadar also documents authenticated `GET /v1/legacy/trains/live-map`, returning an array of train numbers, station coordinates, next stations, distances, and timetable-relative fields. The example has an envelope timestamp but no clear per-train GPS observation timestamp. [Aggregate feed](https://railradar.in/docs/legacy-live-map)
- RailRadar terms describe coordinates and delay indicators as crowd-powered telemetry; independent from Railways, IRCTC and NTES. The contributor GPS license grants rights to RailRadar, not expressly to API consumers for retention, redistribution, export, or model training. [Terms, modified May 2026](https://railradar.in/terms)
- RailRadar pricing page has a developer plans title and indexed sales contact `contact@railradar.in`, but retrieved page text only contained navigation/footer; paid pricing could not be validated. [Pricing](https://railradar.in/pricing)
- RailKit explicitly offers SDK plus REST with API-key authentication, a v1 NTES-backed live route, train history, and station boards. Only docs were read; no NTES-backed request was made. [Docs](https://railkit.in/docs)
- RailKit v2 is described as WIMT-backed, with expanded route points/current location/speed; accepted dates range from today through five days ago. This does not establish the upstream permission chain or measured GPS provenance. [V2 docs](https://railkit.in/docs/live-tracking-v2)
- RailKit train history documents completed station-by-station journeys and arrival/departure delay data; maximum historical retention is unspecified. [History](https://railkit.in/docs/train-history)
- RailKit pricing displayed free 50/month; Pro ₹59/month promotional, 5,000/month and 200/10 min; Advance ₹89/month promotional, 10,000/month and 600/10 min; 100,000-request packs ₹749. Enterprise provides custom limits/private infrastructure/custom SLAs. Promotional availability and tax were not checked. [Pricing](https://railkit.in/pricing)
- Qrail documents API keys, base `https://qrail.in/api`, v3/v4; Free 100/day excludes running status, Pro ₹1,999/year includes running status/live station with 10,000/day, Business ₹4,999/year 50,000/day with priority support/SLA. Allowance resets midnight IST; greater headroom requires contacting `api@qrail.in`. [Developer plans](https://qrail.in/developers)
- IndianRailAPI documents a live-status API-key endpoint with journey date, current station and route actual/scheduled times; its sample is dated 2018 and has `CurrentPosition: null`. No current live success, GPS, freshness or paid quota was established. [Live docs](https://indianrailapi.com/api-collection/live-train-status)

### Inferences
- Workload arithmetic, assuming 2,500 trains continuously and 30 days: 2,500 × 720 polls/day = 1.8 million/day = 54 million/month, approximately 20.83 requests/sec or 1,250/minute. This is a project assumption, not a provider claim.
- If one RailRadar aggregate response covers the fleet, polling it every two minutes is 21,600 requests/month, a 2,500-fold reduction. Coverage, measured freshness, cost accounting and permission to use that legacy feed need validation first. Even this exceeds its free tier.
- RailKit Advance's 600/10 min is 60/min, versus required 1,250/min. Buying 540 × 100k packs alone costs ₹404,460 for 54m requests and does not prove increased rate limits; enterprise is the realistic discussion, not top-ups.
- Qrail Business's 50k/day is 36 times below 1.8m/day; public annual pricing does not imply fleet-scale permission.
- RailRadar's `isActualPosition` alone should not be treated as proof of fresh measured coordinates. Retain source observation time and provenance separately from HTTP fetch time and interpolation.

### Gaps
- No authenticated data request, account creation, paid action, or upstream enquiry call was made. Public docs prove contracts are advertised, not operational reliability.
- None of the inspected providers explicitly established this project's bulk caching, raw retention, downloadable exports, commercial redistribution, or ML training rights. Obtain these in writing with upstream licensing and deletion requirements.
- Need a permitted sample across overnight journeys, delayed/diverted/cancelled services, low-traffic routes and unavailable crowdsourcing; count fresh measured positions separately from schedules/estimates. Require observation timestamps, update SLA, all-India coverage, quotas and operational failure rules.
- Public HTTPS documentation GETs through the search connector succeeded. Direct PowerShell HTTP retrieval of RailRadar/RailKit was blocked by environment socket permissions; no shell live-data responses were verified.

## Do RailYatri, ixigo, Railofy, RailRestro or RapidAPI provide a usable licensed feed?

### Takeaway
RailYatri and ixigo clearly have running-status products and crowdsourcing, but public developer access for a fleet tracker was not established. RapidAPI listings advertise live status, yet a marketplace subscription is not evidence of official Railway licensing or high-volume retention rights.

### Cited Findings
- RailYatri says its location data comes from users' mobile GPS or cell-tower networks and that ETA uses machine learning/historical running patterns; it recommends onboard GPS mode. Claims of minute updates are consumer service descriptions, not API SLAs. [Consumer status FAQ](https://www.railyatri.in/live-train-status?device_type_id=6)
- ixigo's own public train status page advertises running-status and delay information; its official public filing describes crowdsourced offline running status. No developer access terms were established. [Consumer page](https://www.ixigo.com/trains/train-running-status); [ixigo filing](https://rocket.ixigo.com/legal/ixigo_DRHP.pdf)
- RailRestro's primary site markets food delivery plus train timetable/status tools; no public live-data developer grant was located. Food ordering/affiliate integration must not be confused with a running-status data API. [Official site](https://www.railrestro.com/)
- RapidAPI provider `socialtoolhub/irctc-api5` advertises `GET liveTrainStatus`; `ksdeveloper/irctc-api3` advertises a POST spot-your-train endpoint; `GatiMan/irctc27` advertises JSON live train status. These are third-party marketplace listings; the IRCTC name is not proof of government ownership. [socialtoolhub endpoint](https://rapidapi.com/socialtoolhub/api/irctc-api5/playground/apiendpoint_d39b3975-a7aa-498f-95bb-8734c1ddb8d9); [ksdeveloper endpoint](https://rapidapi.com/ksdeveloper/api/irctc-api3/playground/apiendpoint_fc66d8c3-c8c2-408c-8bed-b17d799204ab); [GatiMan listing](https://rapidapi.com/GatiMan/api/irctc27)
- Another commonly referenced listing is `IRCTCAPI/irctc1`; the retrieved official marketplace page yielded only the product title, so endpoint schemas, current quotas/prices and successful response content were unavailable. [Listing](https://rapidapi.com/IRCTCAPI/api/irctc1); [Pricing](https://rapidapi.com/IRCTCAPI/api/irctc1/pricing)
- RapidAPI documentation explains provider-specific subscription plans and overage billing; therefore marketplace billing alone cannot establish an adequate provider cap or contractual upstream data rights. [Subscription documentation](https://docs.rapidapi.com/v1.0/docs/api-pricing)

### Inferences
- Consumer API endpoint discovery would answer technical feasibility, but would not solve permission, predictable high-volume quota, or dataset retention. Treat consumer endpoints as prototype evidence until a licensed data agreement exists.
- RailYatri is worth a direct commercial data inquiry because its documented crowdsourcing addresses position provenance; ixigo is similar. Neither is an immediately documented self-service ingestion choice from evidence collected.
- None of the RapidAPI providers can yet be called a verified credible production source for 54m/month: provenance, freshness and contractual reliability remain missing. Avoid stacking unofficial aggregators as if they were independent confirmation.

### Gaps
- No public RailYatri/ixigo/Railofy/RailRestro live-data developer contract or all-India bulk feed was located. This is a search result limitation, not a claim that private partnerships cannot exist.
- Railofy's historical consumer-status URL gave a search-tool redirect to unrelated content. This may be retrieval corruption or site failure; no conclusion about the company's security/availability should follow without independent verification.
- RapidAPI current price tiers/rate caps and retention/training rights could not be confirmed without access to fuller listing content; no subscription or endpoint request was made.

## Is official RTIS GPS data available for this project?

### Takeaway
Official onboard train GPS exists and has much better defined sampling than commercial running-status pages, but the inspected CRIS documentation does not grant public API access. A Railway/CRIS-approved data agreement is the credible route to official fleet telemetry.

### Cited Findings
- CRIS documents RTIS locomotive NavIC/GAGAN devices transmitting latitude/longitude and speed to a central server every 30 seconds; onboard per-second logs have 15-day retention. This retention is on-device and does not imply public downloadable history. [CRIS RTIS project](https://cris.org.in/loadpage?page=proRTIS)
- The same CRIS page documents arrival/departure/run-through event generation, central-server relay to COA, integration into NTES and other operational systems, 10,000+ covered locomotives and roughly 77% automated passenger control-chart plotting. These are locomotive/operational metrics, not guaranteed coverage of the project's 2,500 train journeys. [CRIS RTIS project](https://cris.org.in/loadpage?page=proRTIS)

### Inferences
- A station event surfaced through a passenger enquiry site may originate from RTIS, but that does not turn the page's estimated moving marker into raw onboard GPS. Preserve this distinction in the tracker and any training data.
- A public claim of official CRIS access from an unrelated commercial domain needs corroboration from CRIS/Railways or a licensing agreement, rather than the reseller's banner.

### Gaps
- No public CRIS/RTIS API specification, signup grant, open live-position license, quotas or tariff was found in inspected official project documentation. No NTES or etrain enquiry endpoint was contacted.
- Official bulk access, latency guarantees, onward storage/training/export permission and fees require a documented licensing route; the RTIS technical capability alone cannot resolve these.
