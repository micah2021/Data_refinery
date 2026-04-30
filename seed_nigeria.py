"""
seed_nigeria.py — Nigeria Health AI Data Refinery: Full Data Seeder
====================================================================
This script:
1. Seeds the `lga` table with all 774 Nigerian LGAs across 36 states + FCT
2. Creates ./data/ncdc_alerts.csv  (NCDC surveillance alerts, 2019–2025)
3. Creates ./data/nimet_climate.csv (NiMET climate records, 2019–2025)

Run ONCE before data_collector.py:
    python seed_nigeria.py

After this, run:
    python data_collector.py
"""

import csv
import os
import random
import sqlite3
from pathlib import Path
from datetime import date, timedelta

random.seed(42)  # reproducible

# ---------------------------------------------------------------------------
# Nigeria LGA data — all 36 states + FCT, 774 LGAs
# ---------------------------------------------------------------------------

STATES = {
    "Abia":          {"zone": "SE",  "lgas": ["Aba North","Aba South","Arochukwu","Bende","Ikwuano","Isiala Ngwa North","Isiala Ngwa South","Isuikwuato","Obi Ngwa","Ohafia","Osisioma","Ugwunagbo","Ukwa East","Ukwa West","Umuahia North","Umuahia South","Umu Nneochi"]},
    "Adamawa":       {"zone": "NE",  "lgas": ["Demsa","Fufure","Ganye","Gayuk","Gombi","Grie","Hong","Jada","Lamurde","Madagali","Maiha","Mayo Belwa","Michika","Mubi North","Mubi South","Numan","Shelleng","Song","Toungo","Yola North","Yola South"]},
    "Akwa Ibom":     {"zone": "SS", "lgas": ["Abak","Eastern Obolo","Eket","Esit Eket","Essien Udim","Etim Ekpo","Etinan","Ibeno","Ibesikpo Asutan","Ibiono-Ibom","Ika","Ikono","Ikot Abasi","Ikot Ekpene","Ini","Itu","Mbo","Mkpat-Enin","Nsit-Atai","Nsit-Ibom","Nsit-Ubium","Obot Akara","Okobo","Onna","Oron","Oruk Anam","Udung-Uko","Ukanafun","Uruan","Urue-Offong/Oruko","Uyo"]},
    "Anambra":       {"zone": "SE",  "lgas": ["Aguata","Anambra East","Anambra West","Anaocha","Awka North","Awka South","Ayamelum","Dunukofia","Ekwusigo","Idemili North","Idemili South","Ihiala","Njikoka","Nnewi North","Nnewi South","Ogbaru","Onitsha North","Onitsha South","Orumba North","Orumba South","Oyi"]},
    "Bauchi":        {"zone": "NE",  "lgas": ["Alkaleri","Bauchi","Bogoro","Damban","Darazo","Dass","Ganjuwa","Giade","Itas/Gadau","Jama'are","Katagum","Kirfi","Misau","Ningi","Shira","Tafawa Balewa","Toro","Warji","Zaki"]},
    "Bayelsa":       {"zone": "SS", "lgas": ["Brass","Ekeremor","Kolokuma/Opokuma","Nembe","Ogbia","Sagbama","Southern Ijaw","Yenagoa"]},
    "Benue":         {"zone": "NC","lgas": ["Ado","Agatu","Apa","Buruku","Gboko","Guma","Gwer East","Gwer West","Katsina-Ala","Konshisha","Kwande","Logo","Makurdi","Obi","Ogbadibo","Ohimini","Oju","Okpokwu","Otukpo","Tarka","Ukum","Ushongo","Vandeikya"]},
    "Borno":         {"zone": "NE",  "lgas": ["Abadam","Askira/Uba","Bama","Bayo","Biu","Chibok","Damboa","Dikwa","Gubio","Guzamala","Gwoza","Hawul","Jere","Kaga","Kala/Balge","Konduga","Kukawa","Kwaya Kusar","Mafa","Magumeri","Maiduguri","Marte","Mobbar","Monguno","Ngala","Nganzai","Shani"]},
    "Cross River":   {"zone": "SS", "lgas": ["Abi","Akamkpa","Akpabuyo","Bakassi","Bekwarra","Biase","Boki","Calabar Municipal","Calabar South","Etung","Ikom","Obanliku","Obubra","Obudu","Odukpani","Ogoja","Yakuur","Yala"]},
    "Delta":         {"zone": "SS", "lgas": ["Aniocha North","Aniocha South","Bomadi","Burutu","Ethiope East","Ethiope West","Ika North East","Ika South","Isoko North","Isoko South","Ndokwa East","Ndokwa West","Okpe","Oshimili North","Oshimili South","Patani","Sapele","Udu","Ughelli North","Ughelli South","Ukwuani","Uvwie","Warri North","Warri South","Warri South West"]},
    "Ebonyi":        {"zone": "SE",  "lgas": ["Abakaliki","Afikpo North","Afikpo South","Ebonyi","Ezza North","Ezza South","Ikwo","Ishielu","Ivo","Izzi","Ohaozara","Ohaukwu","Onicha"]},
    "Edo":           {"zone": "SS", "lgas": ["Akoko-Edo","Egor","Esan Central","Esan North-East","Esan South-East","Esan West","Etsako Central","Etsako East","Etsako West","Igueben","Ikpoba-Okha","Orhionmwon","Oredo","Ovia North-East","Ovia South-West","Owan East","Owan West","Uhunmwonde"]},
    "Ekiti":         {"zone": "SW",  "lgas": ["Ado Ekiti","Efon","Ekiti East","Ekiti South-West","Ekiti West","Emure","Gbonyin","Ido/Osi","Ijero","Ikere","Ikole","Ilejemeje","Irepodun/Ifelodun","Ise/Orun","Moba","Oye"]},
    "Enugu":         {"zone": "SE",  "lgas": ["Aninri","Awgu","Enugu East","Enugu North","Enugu South","Ezeagu","Igbo Etiti","Igbo Eze North","Igbo Eze South","Isi Uzo","Nkanu East","Nkanu West","Nsukka","Oji River","Udenu","Udi","Uzo Uwani"]},
    "FCT":           {"zone": "NC","lgas": ["Abaji","Bwari","Gwagwalada","Kuje","Kwali","Municipal Area Council"]},
    "Gombe":         {"zone": "NE",  "lgas": ["Akko","Balanga","Billiri","Dukku","Funakaye","Gombe","Kaltungo","Kwami","Nafada","Shongom","Yamaltu/Deba"]},
    "Imo":           {"zone": "SE",  "lgas": ["Aboh Mbaise","Ahiazu Mbaise","Ehime Mbano","Ezinihitte","Ideato North","Ideato South","Ihitte/Uboma","Ikeduru","Isiala Mbano","Isu","Mbaitoli","Ngor Okpala","Njaba","Nkwerre","Nwangele","Obowo","Oguta","Ohaji/Egbema","Okigwe","Orlu","Orsu","Oru East","Oru West","Owerri Municipal","Owerri North","Owerri West","Unuimo"]},
    "Jigawa":        {"zone": "NW",  "lgas": ["Auyo","Babura","Biriniwa","Birnin Kudu","Buji","Dutse","Gagarawa","Garki","Gumel","Guri","Gwaram","Gwiwa","Hadejia","Jahun","Kafin Hausa","Kaugama","Kazaure","Kiri Kasama","Maigatari","Malam Madori","Miga","Ringim","Roni","Sule Tankarkar","Taura","Yankwashi"]},
    "Kaduna":        {"zone": "NW",  "lgas": ["Birnin Gwari","Chikun","Giwa","Igabi","Ikara","Jaba","Jema'a","Kachia","Kaduna North","Kaduna South","Kagarko","Kajuru","Kaura","Kauru","Kubau","Kudan","Lere","Makarfi","Sabon Gari","Sanga","Soba","Zangon Kataf","Zaria"]},
    "Kano":          {"zone": "NW",  "lgas": ["Albasu","Bagwai","Bebeji","Bichi","Bunkure","Dala","Dambatta","Dawakin Kudu","Dawakin Tofa","Doguwa","Fagge","Gabasawa","Garko","Garun Mallam","Gaya","Gezawa","Gwale","Gwarzo","Kabo","Kano Municipal","Karaye","Kibiya","Kiru","Kumbotso","Kunchi","Kura","Madobi","Makoda","Minjibir","Nasarawa","Rano","Rimin Gado","Rogo","Shanono","Sumaila","Takai","Tarauni","Tofa","Tsanyawa","Tudun Wada","Ungogo","Warawa","Wudil"]},
    "Katsina":       {"zone": "NW",  "lgas": ["Bakori","Batagarawa","Batsari","Baure","Bindawa","Charanchi","Dandume","Danja","Dan Musa","Daura","Dutsi","Dutsin-Ma","Faskari","Funtua","Ingawa","Jibia","Kafur","Kaita","Kankara","Kankia","Katsina","Kurfi","Kusada","Mai'adua","Malumfashi","Mani","Mashi","Matazu","Musawa","Rimi","Sabuwa","Safana","Sandamu","Zango"]},
    "Kebbi":         {"zone": "NW",  "lgas": ["Aleiro","Arewa Dandi","Argungu","Augie","Bagudo","Birnin Kebbi","Bunza","Dandi","Fakai","Gwandu","Jega","Kalgo","Koko/Besse","Maiyama","Ngaski","Sakaba","Shanga","Suru","Wasagu/Danko","Yauri","Zuru"]},
    "Kogi":          {"zone": "NC","lgas": ["Adavi","Ajaokuta","Ankpa","Bassa","Dekina","Ibaji","Idah","Igalamela-Odolu","Ijumu","Kabba/Bunu","Kogi","Lokoja","Mopa-Muro","Ofu","Ogori/Magongo","Okehi","Okene","Olamaboro","Omala","Yagba East","Yagba West"]},
    "Kwara":         {"zone": "NC","lgas": ["Asa","Baruten","Edu","Ekiti","Ifelodun","Ilorin East","Ilorin South","Ilorin West","Irepodun","Isin","Kaiama","Moro","Offa","Oke Ero","Oyun","Pategi"]},
    "Lagos":         {"zone": "SW",  "lgas": ["Agege","Ajeromi-Ifelodun","Alimosho","Amuwo-Odofin","Apapa","Badagry","Epe","Eti-Osa","Ibeju-Lekki","Ifako-Ijaiye","Ikeja","Ikorodu","Kosofe","Lagos Island","Lagos Mainland","Mushin","Ojo","Oshodi-Isolo","Shomolu","Surulere"]},
    "Nasarawa":      {"zone": "NC","lgas": ["Akwanga","Awe","Doma","Karu","Keana","Keffi","Kokona","Lafia","Nasarawa","Nasarawa Egon","Obi","Toto","Wamba"]},
    "Niger":         {"zone": "NC","lgas": ["Agaie","Agwara","Bida","Borgu","Bosso","Chanchaga","Edati","Gbako","Gurara","Katcha","Kontagora","Lapai","Lavun","Magama","Mariga","Mashegu","Mokwa","Moya","Paikoro","Rafi","Rijau","Shiroro","Suleja","Tafa","Wushishi"]},
    "Ogun":          {"zone": "SW",  "lgas": ["Abeokuta North","Abeokuta South","Ado-Odo/Ota","Egbado North","Egbado South","Ewekoro","Ifo","Ijebu East","Ijebu North","Ijebu North East","Ijebu Ode","Ikenne","Imeko Afon","Ipokia","Obafemi Owode","Odeda","Odogbolu","Ogun Waterside","Remo North","Shagamu"]},
    "Ondo":          {"zone": "SW",  "lgas": ["Akoko North-East","Akoko North-West","Akoko South-East","Akoko South-West","Akure North","Akure South","Ese Odo","Idanre","Ifedore","Ilaje","Ile Oluji/Okeigbo","Irele","Odigbo","Okitipupa","Ondo East","Ondo West","Ose","Owo"]},
    "Osun":          {"zone": "SW",  "lgas": ["Aiyedaade","Aiyedire","Atakumosa East","Atakumosa West","Boripe","Ede North","Ede South","Egbedore","Ejigbo","Ife Central","Ife East","Ife North","Ife South","Ifedayo","Ifelodun","Ila","Ilesa East","Ilesa West","Irepodun","Irewole","Isokan","Iwo","Obokun","Odo-Otin","Ola Oluwa","Olorunda","Oriade","Orolu","Osogbo"]},
    "Oyo":           {"zone": "SW",  "lgas": ["Afijio","Akinyele","Atiba","Atisbo","Egbeda","Ibadan North","Ibadan North-East","Ibadan North-West","Ibadan South-East","Ibadan South-West","Ibarapa Central","Ibarapa East","Ibarapa North","Ido","Irepo","Iseyin","Itesiwaju","Iwajowa","Kajola","Lagelu","Ogbomosho North","Ogbomosho South","Ogo Oluwa","Olorunsogo","Oluyole","Ona Ara","Orelope","Orire","Oyo East","Oyo West","Saki East","Saki West","Surulere"]},
    "Plateau":       {"zone": "NC","lgas": ["Barkin Ladi","Bassa","Bokkos","Jos East","Jos North","Jos South","Kanam","Kanke","Langtang North","Langtang South","Mangu","Mikang","Pankshin","Qua'an Pan","Riyom","Shendam","Wase"]},
    "Rivers":        {"zone": "SS", "lgas": ["Abua/Odual","Ahoada East","Ahoada West","Akuku-Toru","Andoni","Asari-Toru","Bonny","Degema","Eleme","Emuoha","Etche","Gokana","Ikwerre","Khana","Obio/Akpor","Ogba/Egbema/Ndoni","Ogu/Bolo","Okrika","Omuma","Opobo/Nkoro","Oyigbo","Port Harcourt","Tai"]},
    "Sokoto":        {"zone": "NW",  "lgas": ["Binji","Bodinga","Dange Shuni","Gada","Goronyo","Gudu","Gwadabawa","Illela","Isa","Kebbe","Kware","Rabah","Sabon Birni","Shagari","Silame","Sokoto North","Sokoto South","Tambuwal","Tangaza","Tureta","Wamako","Wurno","Yabo"]},
    "Taraba":        {"zone": "NE",  "lgas": ["Ardo Kola","Bali","Donga","Gashaka","Gassol","Ibi","Jalingo","Karim Lamido","Kumi","Lau","Sardauna","Takum","Ussa","Wukari","Yorro","Zing"]},
    "Yobe":          {"zone": "NE",  "lgas": ["Bade","Bursari","Damaturu","Fika","Fune","Geidam","Gujba","Gulani","Jakusko","Karasuwa","Machina","Nangere","Nguru","Potiskum","Tarmuwa","Yunusari","Yusufari"]},
    "Zamfara":       {"zone": "NW",  "lgas": ["Anka","Bakura","Birnin Magaji/Kiyaw","Bukkuyum","Bungudu","Gummi","Gusau","Kaura Namoda","Maradun","Maru","Shinkafi","Talata Mafara","Tsafe","Zurmi"]},
}

ZONE_LGA_TYPES = {
    "NE":   {"rural": 0.75, "urban": 0.15, "semi-urban": 0.10},
    "NW":   {"rural": 0.70, "urban": 0.20, "semi-urban": 0.10},
    "NC":{"rural": 0.65, "urban": 0.20, "semi-urban": 0.15},
    "SE":   {"rural": 0.55, "urban": 0.25, "semi-urban": 0.20},
    "SS":  {"rural": 0.55, "urban": 0.25, "semi-urban": 0.20},
    "SW":   {"rural": 0.40, "urban": 0.40, "semi-urban": 0.20},
}

ZONE_POP_DENSITY = {
    "NE":    (50,  300),
    "NW":    (80,  500),
    "NC": (60,  400),
    "SE":    (200, 900),
    "SS":   (150, 700),
    "SW":    (200, 1200),
}

DISEASES = ["malaria", "cholera", "typhoid", "tuberculosis", "meningitis",
            "lassa_fever", "yellow_fever", "respiratory", "diarrhoeal", "hiv_aids"]

ALERT_LEVELS = ["suspected", "confirmed", "outbreak_declared", "rumour"]

# ---------------------------------------------------------------------------
# Step 1: Seed LGA table
# ---------------------------------------------------------------------------

def seed_lga_table(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    
    inserted = 0
    lga_id = 1
    
    for state, info in STATES.items():
        zone = info["zone"]
        type_dist = ZONE_LGA_TYPES[zone]
        pop_range = ZONE_POP_DENSITY[zone]
        
        for lga_name in info["lgas"]:
            lga_code = f"{state[:3].upper()}-{lga_name[:4].upper().replace(' ', '')}-{lga_id:03d}"
            
            # Assign lga_type based on zone distribution
            r = random.random()
            if r < type_dist["urban"]:
                lga_type = "urban"
            elif r < type_dist["urban"] + type_dist["semi-urban"]:
                lga_type = "semi-urban"
            else:
                lga_type = "rural"
            
            pop_density = round(random.uniform(*pop_range), 1)
            lat = round(random.uniform(4.5, 13.9), 4)
            lng = round(random.uniform(2.7, 14.7), 4)
            
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO lga
                        (lga_id, lga_name, lga_code, state, zone, lga_type, pop_density, lat, lng)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (lga_id, lga_name, lga_code, state, zone, lga_type, pop_density, lat, lng)
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass
            
            lga_id += 1
    
    conn.commit()
    conn.close()
    print(f"  ✓ Seeded {inserted} LGAs into lga table")
    return inserted


# ---------------------------------------------------------------------------
# Step 2: Generate NCDC alerts CSV
# ---------------------------------------------------------------------------

def generate_ncdc_csv(output_path: str, lgas: list[dict]) -> int:
    rows = []
    start = date(2019, 1, 1)
    end = date(2025, 12, 31)
    
    # Generate ~3000 alert records
    for _ in range(3000):
        lga = random.choice(lgas)
        
        # Random date in range
        delta = (end - start).days
        alert_date = start + timedelta(days=random.randint(0, delta))
        
        disease = random.choice(DISEASES)
        alert_level = random.choices(
            ALERT_LEVELS, weights=[0.40, 0.35, 0.10, 0.15]
        )[0]
        
        suspected = random.randint(1, 200)
        confirmed = random.randint(0, suspected) if alert_level in ("confirmed", "outbreak_declared") else 0
        deaths = random.randint(0, max(1, confirmed // 10))
        
        rows.append({
            "alert_date": alert_date.isoformat(),
            "state": lga["state"],
            "lga_name": lga["lga_name"],
            "disease": disease,
            "alert_level": alert_level,
            "suspected_cases": suspected,
            "confirmed_cases": confirmed,
            "deaths": deaths,
            "ncdc_ref": f"NCDC-{alert_date.year}-{random.randint(10000,99999)}",
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ Generated {len(rows)} NCDC alert records → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# Step 3: Generate NiMET climate CSV
# ---------------------------------------------------------------------------

ZONE_CLIMATE = {
    "NE":    {"rain": (20,  400),  "tmax": (32, 42), "tmin": (15, 28), "humidity": (20, 65)},
    "NW":    {"rain": (10,  350),  "tmax": (33, 43), "tmin": (14, 27), "humidity": (15, 60)},
    "NC": {"rain": (50,  900),  "tmax": (30, 40), "tmin": (18, 28), "humidity": (30, 75)},
    "SE":    {"rain": (100, 2500), "tmax": (27, 34), "tmin": (20, 26), "humidity": (60, 95)},
    "SS":   {"rain": (150, 3000), "tmax": (26, 33), "tmin": (20, 26), "humidity": (65, 98)},
    "SW":    {"rain": (80,  1800), "tmax": (27, 35), "tmin": (20, 27), "humidity": (55, 92)},
}

def generate_nimet_csv(output_path: str, lgas: list[dict]) -> int:
    rows = []
    
    # Monthly records 2019–2025 for a sample of LGAs (all 774 × 84 months = 65k rows)
    for lga in lgas:
        zone = lga["zone"]
        climate = ZONE_CLIMATE[zone]
        
        for year in range(2019, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 6:
                    continue  # future data
                
                # Seasonal rainfall adjustment
                if month in (6, 7, 8, 9):  # peak wet
                    rain_factor = 1.0
                elif month in (5, 10):      # shoulder
                    rain_factor = 0.6
                elif month in (3, 4, 11):   # dry start/end
                    rain_factor = 0.2
                else:                        # harmattan
                    rain_factor = 0.05
                
                rain_min = climate["rain"][0] * rain_factor
                rain_max = climate["rain"][1] * rain_factor
                rainfall = round(random.uniform(rain_min, rain_max), 1)
                
                temp_max = round(random.uniform(*climate["tmax"]), 1)
                temp_min = round(random.uniform(*climate["tmin"]), 1)
                humidity = round(random.uniform(*climate["humidity"]), 1)
                
                flood_risk = 1 if (rainfall > climate["rain"][1] * 0.7 and month in range(6, 11)) else 0
                drought = 1 if (rainfall < 30 and month in [11, 12, 1, 2, 3]) else 0
                
                rows.append({
                    "lga_name": lga["lga_name"],
                    "state": lga["state"],
                    "zone": zone,
                    "year": year,
                    "month": month,
                    "rainfall_mm": rainfall,
                    "temp_max_c": temp_max,
                    "temp_min_c": temp_min,
                    "humidity_pct": humidity,
                    "ndvi": round(random.uniform(0.1, 0.8), 3),
                    "flood_risk_flag": flood_risk,
                    "drought_flag": drought,
                })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ Generated {len(rows)} NiMET climate records → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# Step 4: Seed disease_record table directly (replaces broken DHIS2)
# ---------------------------------------------------------------------------

def seed_disease_records(db_path: str, lgas: list[dict]) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    
    rows = []
    start = date(2019, 1, 1)
    end = date(2025, 6, 30)
    
    DISEASE_PROFILES = {
        "malaria":                {"base_cases": (5, 300),  "death_rate": 0.005, "icd10": "B54"},
        "cholera":                {"base_cases": (1, 80),   "death_rate": 0.02,  "icd10": "A00"},
        "typhoid":                {"base_cases": (2, 60),   "death_rate": 0.01,  "icd10": "A01.0"},
        "tuberculosis":           {"base_cases": (1, 30),   "death_rate": 0.03,  "icd10": "A15"},
        "meningitis":             {"base_cases": (0, 20),   "death_rate": 0.08,  "icd10": "G03"},
        "lassa_fever":            {"base_cases": (0, 10),   "death_rate": 0.20,  "icd10": "A96.2"},
        "respiratory":              {"base_cases": (5, 150),  "death_rate": 0.015, "icd10": "J18"},
        "diarrhoeal": {"base_cases": (3, 100),  "death_rate": 0.005, "icd10": "A09"},
    }
    
    print("  Generating disease records (this may take ~30 seconds)...")
    
    inserted = 0
    for lga in lgas:
        # Each LGA gets weekly records for each disease
        for disease, profile in DISEASE_PROFILES.items():
            # Generate one record per epi-week for 2019–2025
            current = start
            while current <= end:
                epi_week = current.isocalendar()[1]
                epi_year = current.isocalendar()[0]
                
                # Seasonal multiplier for malaria/cholera
                month = current.month
                if disease == "malaria" and month in (5, 6, 7, 8, 9, 10):
                    multiplier = 2.5
                elif disease == "cholera" and month in (4, 5, 6, 7, 8):
                    multiplier = 3.0
                else:
                    multiplier = 1.0
                
                base_min, base_max = profile["base_cases"]
                case_count = int(random.randint(base_min, base_max) * multiplier)
                death_count = int(case_count * profile["death_rate"] * random.uniform(0.5, 1.5))
                
                if case_count == 0 and random.random() > 0.3:
                    current += timedelta(weeks=1)
                    continue
                
                rows.append((
                    lga["lga_id"],
                    None,  # facility_id
                    current.isoformat(),
                    epi_week,
                    epi_year,
                    profile["icd10"],
                    disease.replace("_", " "),
                    disease,
                    case_count,
                    death_count,
                    random.choice(["<5", "5-14", "15-44", "45-64", "65+"]),
                    random.choice(["male", "female", "unknown"]),
                    1 if random.random() > 0.6 else 0,
                    round(random.uniform(0.4, 0.95), 3),
                    "synthetic_dhis2",
                    f"SYN-{lga['lga_id']}-{disease[:3].upper()}-{epi_year}W{epi_week:02d}",
                ))
                
                current += timedelta(weeks=1)
                
                # Batch insert
                if len(rows) >= 2000:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO disease_record
                            (lga_id, facility_id, report_date, epi_week, epi_year,
                             icd10_code, disease_name, disease_category, case_count,
                             death_count, age_group, sex, is_confirmed,
                             data_quality_score, source, raw_record_ref)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        rows
                    )
                    conn.commit()
                    inserted += len(rows)
                    rows = []
    
    if rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO disease_record
                (lga_id, facility_id, report_date, epi_week, epi_year,
                 icd10_code, disease_name, disease_category, case_count,
                 death_count, age_group, sex, is_confirmed,
                 data_quality_score, source, raw_record_ref)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows
        )
        conn.commit()
        inserted += len(rows)
    
    conn.close()
    print(f"  ✓ Inserted {inserted:,} disease records into disease_record table")
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db_path = os.getenv("DB_PATH", "./nigeria.db")
    
    print("=" * 60)
    print("Nigeria Health AI — Data Seeder")
    print("=" * 60)
    
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Run: python -c \"import sqlite3; ...\" to create it first")
        return
    
    print(f"\n[1/5] Seeding LGA table in {db_path}...")
    n_lgas = seed_lga_table(db_path)
    
    # Load LGA list for use in CSV generation
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    lgas = [dict(r) for r in conn.execute("SELECT lga_id, lga_name, state, zone, lga_type FROM lga").fetchall()]
    conn.close()
    print(f"  Loaded {len(lgas)} LGAs for data generation")
    
    if not lgas:
        print("ERROR: LGA table still empty — check your schema")
        return
    
    print(f"\n[2/5] Generating NCDC surveillance alerts CSV...")
    generate_ncdc_csv("./data/ncdc_alerts.csv", lgas)
    
    print(f"\n[3/5] Generating NiMET climate CSV...")
    generate_nimet_csv("./data/nimet_climate.csv", lgas)
    
    print(f"\n[4/5] Seeding disease records directly...")
    # Use a sample of LGAs to keep size manageable (all 774 × 8 diseases × 330 weeks = ~2M rows)
    sample_lgas = random.sample(lgas, min(100, len(lgas)))
    seed_disease_records(db_path, sample_lgas)
    
    print(f"\n[5/5] Verifying database...")
    conn = sqlite3.connect(db_path)
    for table in ["lga", "disease_record", "surveillance_alert", "climate_health", "socioeconomic"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:30s}: {count:>10,} rows")
    conn.close()
    
    print("\n" + "=" * 60)
    print("✓ Seeding complete! Now run: python data_collector.py")
    print("  The world_bank, ncdc, and fao collectors will now")
    print("  find data to process.")
    print("=" * 60)


if __name__ == "__main__":
    main()
