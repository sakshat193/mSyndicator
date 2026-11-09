# Goibibo live train status API suitability

## Does the website have a functioning, reusable status API?

### Takeaway
Yes: one normal browser query for train 12951 returned HTTP 200 and `success: true` from a MakeMyTrip rail API. This verifies a functioning single-train running-status call, not a supported public developer API or nationwide feed.

### Cited Findings
- The live-status landing page provides a train-number/name selector and a CHECK STATUS action. Browser observation on 7 October 2026 showed an initial `POST https://mapi.goibibo.com/empeiria/api/v1/desktoprail`; this was page-load traffic, not a verified status query. — [Goibibo status page](https://www.goibibo.com/trains/check-train-running-status/)
- Normal UI query captured `POST https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2` with JSON body `{"trainNumber":"12951","dateOfJourney":"07-10-2026","findNextRunningDate":true}`. It returned HTTP 200, `success: true`. Request used `Content-Type: application/json`, JSON Accept, Goibibo Referer, browser User-Agent and normal browser client hints; captured request headers showed no Cookie or Authorization header. This does not prove cookies/tokens are never required. — [Observed endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2), invoked through [Goibibo](https://www.goibibo.com/trains/check-train-running-status/)
- Companion calls observed HTTP 200: `GET https://ground-auto-suggest.makemytrip.com/rails/autosuggest/trains?search_query=12951&limit=5&version=v3` and `GET https://rails-ris.makemytrip.com/api/ris/platform/locator/v1?train_code=12951&flavour=desktop`. The latter returns route/platform information and intermediate station coordinates. — [Autosuggest endpoint](https://ground-auto-suggest.makemytrip.com/rails/autosuggest/trains?search_query=12951&limit=5&version=v3); [Platform locator endpoint](https://rails-ris.makemytrip.com/api/ris/platform/locator/v1?train_code=12951&flavour=desktop)
- One standalone stdlib `urllib.request.urlopen` request to the public landing page returned HTTP 403. No cookies, token copying, proxy rotation, or protection bypass was attempted. This is an observed outcome for this environment, not proof that every API endpoint rejects standalone callers. — [Tested page](https://www.goibibo.com/trains/check-train-running-status/)
- One independent PowerShell POST to the verified live-status endpoint used the same JSON body and `Content-Type: application/json`, without Cookie, Authorization, custom User-Agent or Referer. It timed out after 25 seconds. Initial sandbox socket denial was environmental; the permitted network retry timed out. Thus browser success is verified, but server-style reproducibility is not. The API did not return a verified HTTP 403. — [Tested endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)
- Stronger browser test: `fetch` to the same endpoint with `credentials: 'omit'`, only an explicit JSON content-type, and `findNextRunningDate: false` returned HTTP 200, `success: true`, `error: null`, train 12951, start date 7 October 2026, current station ST, last updated 19:49 and eight station entries. This confirms cookies and Authorization were not needed in this tested browser-origin request. Browser User-Agent/Origin/Referer were implicit; server reproducibility remains unconfirmed. — [Tested endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)

Observed request (reproduction example; independent server execution not confirmed):

```http
POST /api/ris/train/livestatus/v2 HTTP/1.1
Host: rails-ris.makemytrip.com
Content-Type: application/json

{"trainNumber":"12951","dateOfJourney":"07-10-2026","findNextRunningDate":true}
```

The normal UI sent `true`; the separate cookie-free browser check sent `false` and succeeded for this date.

### Inferences
- Page-specific internal endpoints would require more maintenance than an API with a published contract; an internal POST is not evidence of permission or stability.
- For an ingestion adapter, prefer `findNextRunningDate: false` and validate returned `startDate` against the requested origin date so a fallback does not silently attach another run's status. The exact fallback semantics are undocumented.

### Gaps
- Standalone POST equivalence remains unconfirmed after the timed-out test. A delegated browser session could not access the coordinator's active browser profile (browser already in use); the successful UI query was verified in the coordinator's session.
- No published Goibibo train-status developer API or bulk/all-active-trains API was located.

## Does it provide real train coordinates or only running status?

### Takeaway
The verified payload provides station-based running status. Latitude/longitude observed in the route response identify stations; no measured train GPS coordinate was proven.

### Cited Findings
- The browser-rendered landing page describes last-reported train location, estimated station arrival/departure and platform information; it also discusses GPS-enabled trains. These are website claims, not verified payload fields or measured freshness. — [Goibibo status page](https://www.goibibo.com/trains/check-train-running-status/)
- The page says results list route stations, scheduled arrival/departure, actual arrival/departure and delay for visited stations, and estimates for upcoming stations. — [Goibibo status page](https://www.goibibo.com/trains/check-train-running-status/)
- Verified status envelope keys: `response`, `success`, `error`; response keys: `trainDetails`, `stations`, `metaDetails`, `lastUpdated`, `dates`, `disclaimer`. For train 12951 the observed `startDate` was `07-10-2026 00:00:00`, `currentStation` was `ST` (Surat), `distanceCovered` 263, `totalLateMins` 6, and `lastUpdated` `07-10-2026 19:49:00`, seen around 20:00 IST. This is one observation, not a freshness SLA. — [Observed live-status endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)
- Surat's station entry had `arrived: true`, `actArrTime: 19:49`, but `departed: false` despite populated `actDepTime: 19:50`. Future Vadodara had `arrived: false` despite populated `actArrTime: 21:06`. Therefore populated `act` time fields include estimates and cannot alone establish observed arrival/departure. — [Observed live-status endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)
- `intermediateStations` contains lat/lon associated with station codes; `currentIntermediateStation` was null in the observed live-status payload. — [Observed live-status endpoint](https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2)

### Inferences
- Mapping a last station or deriving an estimated position between stations is distinct from displaying measured train GPS coordinates. Keep that distinction in any adapter and map label.

### Gaps
- No measured train latitude/longitude, GPS provenance, position accuracy, or rail track geometry was established. `lastUpdated` exists but the payload's source observation semantics are not documented.
- No measured coverage/freshness test across Indian trains was conducted.

## Is it suitable for this project's all-India tracker?

### Takeaway
Do not make Goibibo's private website API the primary project dependency without an explicit data/API agreement. Its user agreement restricts public and commercial reuse, and standalone access was blocked in this environment.

### Cited Findings
- User Agreement section 4.2 restricts distribution, exchange, modification, sale or transmission of website material for business, commercial or public purposes; section 3 limits allowed content uses. — [Official User Agreement](https://www.goibibo.com/info/user-agreement/)
- Sections 4.5 and 4.7 reserve termination of access and changes to the website. — [Official User Agreement](https://www.goibibo.com/info/user-agreement/)
- Train Terms of Services describe Goibibo as a booking facilitator using IRCTC; they do not provide a developer data license. — [Official Terms of Services](https://www.goibibo.com/info/terms-of-services/)

### Inferences
- A supported licensed provider fits the all-India live map better. Goibibo is a candidate for partnership discussion or manual comparison, subject to permission.
- A technically successful single-train call would still leave nationwide coverage, poll rate, costs, freshness and redistribution rights unresolved.

### Gaps
- Need a provider-approved API agreement and explicit rights for caching, map display and redistribution before production use.
- No provider contact was made, no NTES/etrain endpoint was contacted, and no project live mode or adapter was changed.
