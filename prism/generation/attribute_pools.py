# prism/generation/attribute_pools.py
"""Curated attribute pools and programmatic skeleton sampler for diverse profile generation.

Solves mode collapse in LLM-only profile generation by pre-sampling structured
demographic skeletons, then letting the LLM fill narrative details around fixed anchors.
"""
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Name pools — diverse origins: Anglo, Hispanic, East Asian, South Asian,
# African, Middle Eastern, Eastern European
# ---------------------------------------------------------------------------

FIRST_NAMES_FEMALE = [
    "Emily", "Sarah", "Jessica", "Ashley", "Lauren",          # Anglo
    "Marisol", "Valentina", "Camila", "Lucia", "Isabella",    # Hispanic
    "Yuki", "Mei", "Hana", "Sakura", "Jia",                   # East Asian
    "Priya", "Ananya", "Kavya", "Deepa", "Nisha",             # South Asian
    "Amina", "Fatima", "Zainab", "Aisha", "Layla",            # Middle Eastern
    "Chioma", "Adaeze", "Ngozi", "Amara", "Zuri",             # African
    "Katarina", "Irina", "Marta", "Daria", "Zofia",           # Eastern European
    "Chloe", "Sienna", "Audrey", "Nadia", "Ingrid",           # Other Western
]

FIRST_NAMES_MALE = [
    "James", "Michael", "David", "Daniel", "Ryan",            # Anglo
    "Carlos", "Javier", "Mateo", "Diego", "Andres",           # Hispanic
    "Hiroshi", "Wei", "Tao", "Jin", "Kenji",                  # East Asian
    "Arjun", "Vikram", "Rohan", "Sanjay", "Aditya",           # South Asian
    "Omar", "Hassan", "Khalil", "Tariq", "Yusuf",             # Middle Eastern
    "Kwame", "Emeka", "Sekou", "Tendai", "Jabari",            # African
    "Nikolai", "Andrei", "Marek", "Tomasz", "Pavel",           # Eastern European
    "Liam", "Sebastian", "Felix", "Remy", "Soren",            # Other Western
]

FIRST_NAMES_NEUTRAL = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey",
    "Riley", "Quinn", "Avery", "Skyler", "Sage",
]

LAST_NAMES = [
    # Anglo / Western
    "Thompson", "Mitchell", "Anderson", "Campbell", "Stewart",
    "O'Brien", "Fischer", "Larsson", "Bennett", "Clarke",
    # Hispanic
    "Gutierrez", "Rivera", "Navarro", "Delgado", "Morales",
    "Castillo", "Fernandez", "Soto", "Vega", "Herrera",
    # East Asian
    "Tanaka", "Chen", "Kim", "Nguyen", "Zhang",
    "Sato", "Liu", "Park", "Yamamoto", "Wu",
    # South Asian
    "Sharma", "Patel", "Gupta", "Khan", "Reddy",
    "Desai", "Nair", "Mehta", "Singh", "Chopra",
    # African
    "Okafor", "Mensah", "Diallo", "Mwangi", "Banda",
    "Adeyemi", "Nkomo", "Toure", "Okello", "Dlamini",
    # Middle Eastern
    "Al-Rashid", "Nazari", "Haddad", "Khoury", "Mansour",
    # Eastern European
    "Kowalski", "Ivanova", "Petrovic", "Volkov", "Novak",
]

# ---------------------------------------------------------------------------
# Location pools — ~80% US, ~20% international
# ---------------------------------------------------------------------------

LOCATIONS_US = [
    "Seattle, Washington",
    "Portland, Oregon",
    "San Francisco, California",
    "Los Angeles, California",
    "San Diego, California",
    "Phoenix, Arizona",
    "Denver, Colorado",
    "Austin, Texas",
    "Houston, Texas",
    "Dallas, Texas",
    "Minneapolis, Minnesota",
    "Chicago, Illinois",
    "Detroit, Michigan",
    "Nashville, Tennessee",
    "Atlanta, Georgia",
    "Miami, Florida",
    "Tampa, Florida",
    "Charlotte, North Carolina",
    "Raleigh, North Carolina",
    "Washington, D.C.",
    "Philadelphia, Pennsylvania",
    "Pittsburgh, Pennsylvania",
    "Boston, Massachusetts",
    "New York City, New York",
    "Baltimore, Maryland",
    "Richmond, Virginia",
    "New Orleans, Louisiana",
    "Salt Lake City, Utah",
    "Boise, Idaho",
    "Anchorage, Alaska",
    "Honolulu, Hawaii",
    "Rural Montana",
]

LOCATIONS_INTERNATIONAL = [
    "London, United Kingdom",
    "Toronto, Canada",
    "Mexico City, Mexico",
    "Berlin, Germany",
    "Tokyo, Japan",
    "Mumbai, India",
    "Lagos, Nigeria",
    "Sao Paulo, Brazil",
]

# ---------------------------------------------------------------------------
# Occupation pools — broad spectrum
# ---------------------------------------------------------------------------

OCCUPATIONS = [
    # Healthcare
    "Registered nurse (emergency department)",
    "Physical therapist",
    "Dental hygienist",
    "Pharmacy technician",
    "Home health aide",
    # Tech
    "Software engineer",
    "Data analyst",
    "IT systems administrator",
    "UX designer",
    "Cybersecurity analyst",
    # Trades / Blue-collar
    "Electrician",
    "HVAC technician",
    "Commercial truck driver",
    "Construction foreman",
    "Auto mechanic",
    # Education
    "High school teacher (math)",
    "Elementary school teacher",
    "University adjunct professor",
    "School counselor",
    # Business / White-collar
    "Accountant (CPA)",
    "Marketing manager",
    "Human resources specialist",
    "Financial advisor",
    "Real estate agent",
    "Insurance claims adjuster",
    # Service industry
    "Restaurant general manager",
    "Hotel front desk supervisor",
    "Barista / shift lead at a coffee chain",
    "Rideshare driver (full-time)",
    # Creative / Arts
    "Freelance graphic designer",
    "Photographer (weddings and events)",
    "Music teacher (private studio)",
    "Journalist (local newspaper)",
    # Government / Legal / Nonprofit
    "Social worker (child protective services)",
    "Paralegal",
    "City planning analyst",
    "Nonprofit program coordinator",
    "Police officer",
    # Science / Research
    "Environmental scientist",
    "Clinical research coordinator",
    "Lab technician (biotech)",
    # Other
    "Small business owner (bakery)",
    "Veterinary technician",
    "Personal trainer",
    "Flight attendant",
    "Warehouse operations manager",
    "Stay-at-home parent (returning to workforce)",
    "Retired military (part-time consultant)",
    "Graduate student (PhD candidate)",
]

# ---------------------------------------------------------------------------
# Life situation archetypes — the "drama seed" that drives cross-domain tension
# ---------------------------------------------------------------------------

LIFE_SITUATIONS = [
    "Going through a divorce with custody complications",
    "Recently immigrated and adjusting to a new country",
    "Empty nester struggling with identity after kids left home",
    "New parent dealing with postpartum challenges and career pressure",
    "Mid-career change after being laid off from a long-term job",
    "Managing a chronic illness diagnosis received in the past year",
    "Primary caregiver for an aging parent with dementia",
    "Recovering from addiction (alcohol) and rebuilding relationships",
    "Dealing with workplace harassment and considering legal action",
    "Starting a small business while still employed full-time",
    "Navigating a long-distance relationship with plans to relocate",
    "Returning to school as a mature student while working",
    "Recovering financially after bankruptcy or major debt crisis",
    "Closeted about sexual orientation in a conservative community",
    "Processing grief after the recent death of a close family member",
    "Facing a job performance review / possible termination",
    "Living with a disability and advocating for accommodations",
    "Whistleblower situation at work — knows about ethical violations",
    "Recently promoted to management and struggling with new role",
    "Dealing with a teenager's behavioral issues and school problems",
    "Recovering from a serious accident with ongoing physical therapy",
    "In an emotionally abusive relationship and planning to leave",
    "Experiencing housing instability / at risk of eviction",
    "Competitive athlete balancing training with full-time work",
    "Preparing for retirement within the next 1-2 years",
]

# ---------------------------------------------------------------------------
# Health condition pools — age-appropriate groupings
# ---------------------------------------------------------------------------

HEALTH_CONDITIONS_YOUNG = [  # 19-35
    "Anxiety disorder",
    "ADHD (diagnosed in adulthood)",
    "Seasonal allergies",
    "Sports injury (ACL tear, shoulder)",
    "Acne / dermatitis",
    "Asthma",
    "Eating disorder (in recovery)",
    "Iron deficiency anemia",
    "Migraines",
    "Polycystic ovary syndrome (PCOS)",
]

HEALTH_CONDITIONS_MIDDLE = [  # 36-55
    "Type 2 diabetes",
    "Hypertension",
    "High cholesterol",
    "Chronic back pain",
    "Anxiety / depression",
    "Sleep apnea",
    "Thyroid disorder",
    "Acid reflux / GERD",
    "Endometriosis",
    "Early-stage arthritis",
]

HEALTH_CONDITIONS_OLDER = [  # 56-75
    "Type 2 diabetes",
    "Hypertension",
    "Osteoarthritis",
    "Heart disease / prior cardiac event",
    "Chronic kidney disease (early stage)",
    "Glaucoma",
    "COPD",
    "Depression (late-onset)",
    "Prostate issues",
    "Osteoporosis",
]

# ---------------------------------------------------------------------------
# Sensitivity distribution templates — ensures not every profile is all-5s
# ---------------------------------------------------------------------------

SENSITIVITY_TEMPLATES = [
    # (health, finance, social, schedule, professional, lifestyle)
    {"health": 5, "finance": 4, "social": 4, "schedule": 3, "professional": 3, "lifestyle": 4},
    {"health": 3, "finance": 5, "social": 4, "schedule": 2, "professional": 4, "lifestyle": 3},
    {"health": 4, "finance": 3, "social": 5, "schedule": 4, "professional": 3, "lifestyle": 5},
    {"health": 2, "finance": 4, "social": 3, "schedule": 2, "professional": 5, "lifestyle": 4},
    {"health": 5, "finance": 5, "social": 3, "schedule": 4, "professional": 2, "lifestyle": 3},
    {"health": 3, "finance": 2, "social": 5, "schedule": 3, "professional": 4, "lifestyle": 5},
    {"health": 4, "finance": 4, "social": 2, "schedule": 3, "professional": 5, "lifestyle": 3},
    {"health": 4, "finance": 3, "social": 4, "schedule": 5, "professional": 3, "lifestyle": 4},
    {"health": 5, "finance": 3, "social": 4, "schedule": 3, "professional": 4, "lifestyle": 2},
    {"health": 3, "finance": 5, "social": 5, "schedule": 2, "professional": 3, "lifestyle": 4},
]

# ---------------------------------------------------------------------------
# Age bucket definitions with sampling weights
# ---------------------------------------------------------------------------

AGE_BUCKETS = [
    (19, 25, 0.15),   # young adult
    (26, 35, 0.25),   # early career
    (36, 45, 0.25),   # mid career
    (46, 55, 0.15),   # established
    (56, 65, 0.12),   # late career
    (66, 75, 0.08),   # retired / semi-retired
]


def _weighted_choice(items_with_weights: list[tuple], rng: random.Random):
    """Pick from a list of (item..., weight) tuples using the given RNG."""
    items = items_with_weights
    weights = [w for *_, w in items]
    return rng.choices(items, weights=weights, k=1)[0]


def _get_health_pool(age: int) -> list[str]:
    """Return age-appropriate health condition pool."""
    if age <= 35:
        return HEALTH_CONDITIONS_YOUNG
    elif age <= 55:
        return HEALTH_CONDITIONS_MIDDLE
    else:
        return HEALTH_CONDITIONS_OLDER


def sample_skeleton(
    index: int,
    total: int,
    seed: Optional[int] = None,
) -> dict:
    """Sample a demographic skeleton ensuring diversity across a batch.

    Uses stratified sampling: distributes age buckets, locations, and genders
    evenly across the batch, then fills remaining attributes randomly.

    Args:
        index: 0-based position in the batch (used for stratification).
        total: Total number of profiles to generate in this batch.
        seed: Optional random seed. Defaults to index for reproducibility.

    Returns:
        A skeleton dict with fixed demographic attributes for the LLM to build on.
    """
    rng = random.Random(seed if seed is not None else index + 42)

    # --- Gender (stratified) ---
    gender_cycle = ["Female", "Male", "Female", "Male", "Non-binary"]
    gender = gender_cycle[index % len(gender_cycle)]

    # --- Age (stratified across buckets) ---
    bucket_index = index % len(AGE_BUCKETS)
    low, high, _ = AGE_BUCKETS[bucket_index]
    age = rng.randint(low, high)

    # --- Name (based on gender, sampled without replacement intent) ---
    if gender == "Female":
        first_name = rng.choice(FIRST_NAMES_FEMALE)
    elif gender == "Male":
        first_name = rng.choice(FIRST_NAMES_MALE)
    else:
        first_name = rng.choice(FIRST_NAMES_NEUTRAL)
    last_name = rng.choice(LAST_NAMES)

    # --- Location (stratified: ~80% US, ~20% international) ---
    if index % 5 == 4:  # every 5th profile is international
        location = rng.choice(LOCATIONS_INTERNATIONAL)
    else:
        location = rng.choice(LOCATIONS_US)

    # --- Occupation (round-robin with shuffle for variety) ---
    occupation = OCCUPATIONS[index % len(OCCUPATIONS)]

    # --- Life situation (round-robin with offset) ---
    life_situation = LIFE_SITUATIONS[index % len(LIFE_SITUATIONS)]

    # --- Health conditions (age-appropriate, 1-3 conditions) ---
    health_pool = _get_health_pool(age)
    n_conditions = rng.randint(1, min(3, len(health_pool)))
    health_conditions = rng.sample(health_pool, n_conditions)

    # --- Sensitivity template ---
    sensitivity = SENSITIVITY_TEMPLATES[index % len(SENSITIVITY_TEMPLATES)]

    return {
        "name": f"{first_name} {last_name}",
        "age": age,
        "gender": gender,
        "location": location,
        "occupation": occupation,
        "life_situation": life_situation,
        "health_conditions": health_conditions,
        "sensitivity_targets": sensitivity,
    }


def sample_skeletons(total: int, seed: int = 42) -> list[dict]:
    """Sample a full batch of diverse skeletons.

    Shuffles occupation and life-situation pools first so round-robin
    ordering doesn't always yield the same sequence across runs.

    Args:
        total: Number of skeletons to generate.
        seed: Master seed for reproducibility.

    Returns:
        List of skeleton dicts.
    """
    # Shuffle pools for this batch so repeated runs with same total differ
    rng = random.Random(seed)
    occupations_shuffled = OCCUPATIONS[:]
    rng.shuffle(occupations_shuffled)
    situations_shuffled = LIFE_SITUATIONS[:]
    rng.shuffle(situations_shuffled)

    skeletons = []
    for i in range(total):
        skel = sample_skeleton(i, total, seed=seed + i)
        # Override with shuffled round-robin
        skel["occupation"] = occupations_shuffled[i % len(occupations_shuffled)]
        skel["life_situation"] = situations_shuffled[i % len(situations_shuffled)]
        skeletons.append(skel)

    return skeletons
