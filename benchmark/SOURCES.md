# Sources of the open-data tables in `data_open/`

Sixteen evaluation datasets were built from public-agency and open-licence sources. The tables in `data_open/` are the
exact tables used in the paper: the features are the columns of each file, the last column is the target, and sources
above 100,000 rows are reduced to a stratified sample of 100,000 rows. Features describe the unit or its circumstances
as known before the outcome. Coded categories (for example airport, operator, species, business type, local authority,
equipment maker, and hazardous-material number), location coordinates, and the shift start or time of day are kept as
features; record identifiers, names, street addresses, calendar dates of the event, free text, and every outcome or
consequence field are left out. Unless stated otherwise, the source files were retrieved on 23 October 2026.

| Dataset | Source (publisher) | Unit and target | Licence |
|---|---|---|---|
| cfpb-complaints | Consumer Complaint Database, Consumer Financial Protection Bureau (https://consumerfinance.gov/data-research/consumer-complaints) | one complaint received in 2025; target: timely response (yes/no) | U.S. government work, public domain |
| chicago-crashes | Traffic Crashes – Crashes, City of Chicago Data Portal (https://data.cityofchicago.org, 85ca-t3if) | one crash of 2023; target: most severe injury (5 levels); injury counts, crash type, damage, and report fields removed | City of Chicago Data Portal terms of use |
| cqc-ratings | Latest ratings, 1 October 2026, Care Quality Commission (https://cqc.org.uk/about-us/transparency/using-cqc-data) | one location, Overall/Overall row; target: latest rating (4 levels) | Open Government Licence v3.0 |
| dol-perm | PERM disclosure data FY2025, Office of Foreign Labor Certification, U.S. Department of Labor (https://dol.gov/agencies/eta/foreign-labor/performance); retrieved 24 October 2026 | one case; target: certified or denied (withdrawn cases have no target); 28 application-time fields | U.S. government work, public domain |
| faa-wildlife | National Wildlife Strike Database, Federal Aviation Administration (https://wildlife.faa.gov) | one strike report, 1990–2026; target: indicated damage | U.S. government work, public domain |
| fhrs-ratings | Food Hygiene Rating Scheme open data, Food Standards Agency (https://ratings.food.gov.uk/open-data) | one business with a rating of 0–5 (England, Wales, Northern Ireland); target: rating | Open Government Licence v3.0 |
| maib-occurrences | Marine occurrences, Marine Accident Investigation Branch data portal (https://maps.dft.gov.uk/maib-data-portal) | one occurrence, 2020–2026; target: occurrence severity; main-event fields removed because severity classes are partly defined by them | Open Government Licence v3.0 |
| msha-accidents | Accidents data set, Mine Safety and Health Administration open government data | one accident, injury, or illness report, 2000–2026; target: fatality (degree of injury); rows without a recorded degree removed | U.S. government work, public domain |
| noaa-storm-2025 | Storm Events Database, details 2025, NOAA National Centers for Environmental Information (https://ncei.noaa.gov/stormevents) | one event of 2025; target: at least one direct or indirect death | U.S. government work, public domain |
| ntsb-events | Aviation accident database (avall), National Transportation Safety Board (https://data.ntsb.gov/avdata) | one event, 2008–2026; target: accident or incident | U.S. government work, public domain |
| nyc-restaurants | DOHMH New York City Restaurant Inspection Results (https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j), snapshot of 23 October 2026 | one restaurant at its latest graded inspection; target: grade A/B/C | NYC Open Data terms of use |
| ofsted-childcare | Childcare providers and inspections, most recent inspections at 30 June 2025, Ofsted (gov.uk statistical data set) | one provider with a full inspection; target: overall effectiveness (1–4) | Open Government Licence v3.0 |
| phmsa-hazmat | Hazardous materials incident reports (Form 5800.1), PHMSA, as published by the Data Liberation Project (https://github.com/data-liberation-project/phmsa-hazmat-incident-reports) | first row per report number, 1971–2025; target: hazmat fatality indicator | U.S. government work, public domain |
| sfts-mortality | S1 Data of a hospital cohort with severe fever with thrombocytopenia syndrome, 2017–2024, PLOS figshare (https://api.figshare.com/v2/articles/29130855); retrieved 24 October 2026 | one patient (baseline sheet); target: in-hospital death; ICU transfer and length of stay removed | CC BY 4.0 |
| tuandromd | TUANDROMD (Tezpur University Android malware dataset), UCI Machine Learning Repository, dataset 855 (https://archive.ics.uci.edu/dataset/855) | one application; target: malware or goodware; 241 permission and API features | CC BY 4.0 |
| xylella-puglia | Xylella fastidiosa monitoring data, campaigns 2020–2022, Regione Puglia open data (https://dati.puglia.it); retrieved 24 October 2026 | one sample after removing exact duplicates; target: test result (positive/negative); species, cultivar, symptom at sampling, province | CC BY 4.0 |

The remaining 37 datasets are loaded from their original repositories by `datasets.py`: 17 from OpenML (data ids in
`datasets.json`), 13 from the UCI Machine Learning Repository (dataset ids), and 7 from Kaggle, which are not redistributed here
and are downloaded under the licence stated on each dataset page; the Kaggle dataset names and files are listed in
`datasets.json`.
