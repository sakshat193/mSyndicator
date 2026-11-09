# Goibibo works; licensed feeds fit better

**Goibibo has a functioning single-train running-status API**, verified in its browser on 7 October 2026. It returns station events, delay and estimated times; the observed coordinates identify stations, not measured train GPS. A separate browser request succeeded with credentials omitted, but a standalone server-style POST timed out. This establishes technical access in one browser context, not reliable backend access or permission for a fleet tracker. **Prioritize RailRadar's documented aggregate feed**, then enterprise access to RailKit or Qrail; RailYatri/ixigo partnerships and official RTIS access are further options. Alternative APIs below were assessed from documentation, without authenticated response tests.

## Goibibo's browser call works; backend reuse remains unverified

The normal Goibibo CHECK STATUS flow called this endpoint and returned HTTP 200 with `success: true` ([Goibibo](https://www.goibibo.com/trains/check-train-running-status/), [observed endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)):

```http
POST https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2
Content-Type: application/json

{"trainNumber":"12951","dateOfJourney":"07-10-2026","findNextRunningDate":true}
```

A second browser `fetch`, with `credentials: 'omit'`, JSON content-type and `findNextRunningDate: false`, also returned HTTP 200, `success: true`, `error: null`, train 12951, origin date 7 October, current station `ST` and eight station entries. **Cookies and Authorization were unnecessary for that tested request**; browser Origin, Referer and User-Agent remained implicit. A separate PowerShell POST timed out after 25 seconds. The public landing page separately returned 403 to Python urllib; this was **not an API 403**. Backend reproducibility therefore remains unconfirmed. The false flag worked for this sample; fallback semantics are undocumented, so a future adapter must validate the returned origin date.

Companion [train autosuggest](https://ground-auto-suggest.makemytrip.com/rails/autosuggest/trains?search_query=12951&limit=5&version=v3) and [platform/route locator](https://rails-ris.makemytrip.com/api/ris/platform/locator/v1?train_code=12951&flavour=desktop) calls returned 200. No supported public developer contract or bulk feed was located. Goibibo's agreement restricts public/commercial reuse of website material; successful HTTP requests do not supply that permission ([User Agreement](https://www.goibibo.com/info/user-agreement/)).

## Station events cannot establish moving GPS positions

The observed payload had `currentStation: ST`, `distanceCovered: 263`, `totalLateMins: 6` and `lastUpdated: 07-10-2026 19:49:00`, seen around 20:00 IST: approximately eleven minutes old in one sample. Its timestamp semantics and freshness SLA remain undocumented. Surat had `arrived: true` and `actArrTime: 19:49`, but `departed: false` despite `actDepTime: 19:50`. Future Vadodara had `arrived: false` despite populated `actArrTime: 21:06`. **Populated “act” fields include estimates; use the arrival/departure flags** ([observed payload endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)).

Route latitudes/longitudes belong to station codes; `currentIntermediateStation` was null. No measured train coordinate, accuracy or GPS provenance was established. The tracker must preserve source observation time separately from fetch time and distinguish measured locations, station reports and interpolated positions.

## Ranked alternatives need fleet-scale contracts and freshness proof

**First: RailRadar**, because the project already has an offline integration and its documented `GET /v1/legacy/trains/live-map` offers an aggregate response. Its authenticated single-train endpoint advertises `lastUpdatedAt`, `isActualPosition`, segment progress, speed and bearing; the aggregate example lacks a clear per-train GPS observation timestamp. These are documentation examples, not tested responses. The free allowance is 1,000/month; paid pricing was not retrievable. Terms describe crowd-powered telemetry but do not clearly grant API consumers retention, redistribution or training rights ([live status](https://railradar.in/docs/live-train-status), [aggregate feed](https://railradar.in/docs/legacy-live-map), [access](https://railradar.in/docs), [terms](https://railradar.in/terms)).

**Second: RailKit**, for documented live tracking and journey history. V2 advertises WIMT-backed location/speed; that does not prove measured GPS or upstream licensing. Advertised promotional plans are ₹59/month for 5,000 requests and ₹89/month for 10,000; Advance permits 600 requests/10 minutes. Enterprise/custom limits are the relevant option ([V2](https://railkit.in/docs/live-tracking-v2), [history](https://railkit.in/docs/train-history), [pricing](https://railkit.in/pricing)). **Third: Qrail**, whose advertised Pro ₹1,999/year offers 10,000/day and Business ₹4,999/year offers 50,000/day; free access excludes running status ([developer plans](https://qrail.in/developers)). Prices and service availability are advertised, unverified offers.

**Fourth: RailYatri/ixigo partnerships.** RailYatri describes user GPS/cell-tower positions and predicted ETAs; ixigo documents crowdsourced running status. Public bulk developer terms were not established ([RailYatri](https://www.railyatri.in/live-train-status?device_type_id=6), [ixigo filing](https://rocket.ixigo.com/legal/ixigo_DRHP.pdf)). **Fifth: official RTIS access** has the strongest documented physical provenance: locomotive NavIC/GAGAN telemetry every 30 seconds. CRIS documents 10,000+ locomotives, not guaranteed coverage of this project's journeys; no public API grant was found ([CRIS](https://cris.org.in/loadpage?page=proRTIS)). RapidAPI listings and IndianRailAPI remain lower-confidence options without verified freshness, capacity or licensing.

## Aggregate coverage changes the project's economics

Under the illustrative assumption of 2,500 trains polled every 120 seconds continuously for 30 days, individual calls require **1.8 million/day, 54 million/month and 1,250/minute**. Qrail Business is 36 times below that daily load; RailKit Advance permits 60/minute. If one aggregate response covers the fleet, the same cadence requires **21,600 requests/month**. That reduction depends on usable coverage, freshness and provider billing rules; it is not a validated RailRadar capacity claim.

## Conclusion

Keep existing offline ingestion and seek a RailRadar fleet-feed agreement first. Production use remains pending written permission covering upstream provenance, quotas, caching, raw retention, public map display, exports and any model training. A permitted sample must establish per-train observation timestamps, measured-versus-estimated coverage and behavior on overnight, delayed, diverted and cancelled journeys before committing to a provider. No account, purchase, provider contact, NTES/etrain enquiry, live-mode change or implementation change was made during this research.
