import os
import time
import requests
import pandas as pd
from datetime import date
from pathlib import Path


# ============================================================
# Config
# ============================================================

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

BASE_URL = "https://openrouter.ai/api/v1"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "openrouter-daily-cost-tracker"
}

# Providers que quieres monitorear.
# Puedes agregar o quitar aliases.
TARGET_PROVIDERS = {
    "DigitalOcean": [
        "digitalocean",
        "digital ocean"
    ],
    "Weights & Biases / CoreWeave": [
        "wandb",
        "w&b",
        "weights & biases",
        "weights and biases",
        "coreweave",
        "core weave"
    ],
    "NovitaAI": [
        "novita",
        "novitaai",
        "novita ai"
    ],
    "DeepInfra": [
        "deepinfra",
        "deep infra"
    ],
    "Together": [
        "together",
        "together ai",
        "together.ai"
    ],
    "Nebius": [
        "nebius",
        "nebius token factory",
        "token factory"
    ]
}

# Costo promedio:
# 50% input tokens + 50% output tokens.
# Puedes cambiar estos pesos si tu uso real es diferente.
INPUT_WEIGHT = 0.5
OUTPUT_WEIGHT = 0.5

REQUEST_SLEEP_SECONDS = 0.15

SNAPSHOT_DATE = date.today().isoformat()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HISTORICAL_CSV = DATA_DIR / "openrouter_provider_model_cost_history.csv"
DAILY_CSV = DATA_DIR / "latest_openrouter_provider_model_costs.csv"
SUMMARY_CSV = DATA_DIR / "openrouter_provider_daily_summary.csv"


# ============================================================
# Helpers
# ============================================================

def get_json(url, params=None):
    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=60
    )

    if response.status_code != 200:
        print("URL:", response.url)
        print("Status code:", response.status_code)
        print("Response text:", response.text[:1500])
        response.raise_for_status()

    return response.json()


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_get_price(pricing, key):
    if isinstance(pricing, dict):
        return safe_float(pricing.get(key))
    return None


def split_model_id(model_id):
    parts = str(model_id).split("/", 1)

    if len(parts) != 2:
        return None, None

    return parts[0], parts[1]


def normalize_provider_text(text):
    return (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def extract_provider_name(endpoint):
    possible_keys = [
        "provider_name",
        "provider",
        "name",
        "slug",
        "provider_slug"
    ]

    for key in possible_keys:
        value = endpoint.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    provider_obj = endpoint.get("provider")

    if isinstance(provider_obj, dict):
        for key in possible_keys:
            value = provider_obj.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def match_target_provider(provider_name):
    if provider_name is None:
        return None

    provider_norm = normalize_provider_text(provider_name)

    for canonical_name, aliases in TARGET_PROVIDERS.items():
        for alias in aliases:
            alias_norm = normalize_provider_text(alias)

            if alias_norm in provider_norm:
                return canonical_name

    return None


def calculate_cost_per_1m_tokens(prompt_price, completion_price):
    if prompt_price is None or completion_price is None:
        return None

    blended_price_per_token = (
        prompt_price * INPUT_WEIGHT
        + completion_price * OUTPUT_WEIGHT
    )

    return blended_price_per_token * 1_000_000


def clean_model_label(model_id, model_name):
    label = str(model_name)

    if ":" in label:
        label = label.split(":", 1)[1].strip()

    if not label or label.lower() == "nan":
        _, slug = split_model_id(model_id)
        label = slug if slug else str(model_id)

    return label


# ============================================================
# Main extraction
# ============================================================

def fetch_daily_costs():
    models_url = f"{BASE_URL}/models"
    models_json = get_json(models_url)

    models_df = pd.DataFrame(models_json.get("data", []))

    if models_df.empty:
        raise ValueError("No se encontraron modelos en OpenRouter /models.")

    rows = []

    for _, model_row in models_df.iterrows():
        model_id = model_row.get("id")
        model_name = model_row.get("name")
        model_label = clean_model_label(model_id, model_name)

        author, slug = split_model_id(model_id)

        if not author or not slug:
            continue

        endpoint_url = f"{BASE_URL}/models/{author}/{slug}/endpoints"

        try:
            endpoint_json = get_json(endpoint_url)
        except Exception as e:
            print(f"Error leyendo endpoints para {model_id}: {e}")
            continue

        endpoints = endpoint_json.get("data", {}).get("endpoints", [])

        if not isinstance(endpoints, list):
            continue

        model_level_pricing = model_row.get("pricing", {})
        model_prompt_price = safe_get_price(model_level_pricing, "prompt")
        model_completion_price = safe_get_price(model_level_pricing, "completion")

        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue

            raw_provider_name = extract_provider_name(endpoint)
            target_provider = match_target_provider(raw_provider_name)

            if target_provider is None:
                continue

            endpoint_pricing = endpoint.get("pricing", {})

            endpoint_prompt_price = safe_get_price(endpoint_pricing, "prompt")
            endpoint_completion_price = safe_get_price(endpoint_pricing, "completion")

            prompt_price_per_token = (
                endpoint_prompt_price
                if endpoint_prompt_price is not None
                else model_prompt_price
            )

            completion_price_per_token = (
                endpoint_completion_price
                if endpoint_completion_price is not None
                else model_completion_price
            )

            cost_per_1m_tokens = calculate_cost_per_1m_tokens(
                prompt_price_per_token,
                completion_price_per_token
            )

            rows.append({
                "snapshot_date": SNAPSHOT_DATE,
                "provider": target_provider,
                "raw_provider_name": raw_provider_name,
                "model_id": model_id,
                "model_name": model_name,
                "model_label": model_label,
                "prompt_price_per_token": prompt_price_per_token,
                "completion_price_per_token": completion_price_per_token,
                "prompt_cost_per_1m_tokens": (
                    prompt_price_per_token * 1_000_000
                    if prompt_price_per_token is not None
                    else None
                ),
                "completion_cost_per_1m_tokens": (
                    completion_price_per_token * 1_000_000
                    if completion_price_per_token is not None
                    else None
                ),
                "avg_cost_per_1m_tokens": cost_per_1m_tokens,
                "input_weight": INPUT_WEIGHT,
                "output_weight": OUTPUT_WEIGHT
            })

        time.sleep(REQUEST_SLEEP_SECONDS)

    daily_df = pd.DataFrame(rows)

    if daily_df.empty:
        raise ValueError("No se encontraron datos para los providers objetivo.")

    daily_df["avg_cost_per_1m_tokens"] = pd.to_numeric(
        daily_df["avg_cost_per_1m_tokens"],
        errors="coerce"
    )

    daily_df = daily_df.dropna(subset=["avg_cost_per_1m_tokens"])

    daily_df = daily_df.drop_duplicates(
        subset=[
            "snapshot_date",
            "provider",
            "raw_provider_name",
            "model_id",
            "prompt_price_per_token",
            "completion_price_per_token"
        ]
    )

    return daily_df


def update_historical_csv(daily_df):
    if HISTORICAL_CSV.exists():
        historical_df = pd.read_csv(HISTORICAL_CSV)

        # Evita duplicar el mismo día si corres el workflow manualmente.
        historical_df = historical_df[
            historical_df["snapshot_date"] != SNAPSHOT_DATE
        ]

        final_df = pd.concat(
            [historical_df, daily_df],
            ignore_index=True
        )
    else:
        final_df = daily_df.copy()

    final_df = final_df.sort_values(
        [
            "snapshot_date",
            "provider",
            "model_id",
            "raw_provider_name"
        ]
    )

    final_df.to_csv(HISTORICAL_CSV, index=False)

    return final_df


def update_daily_files(daily_df):
    daily_df.to_csv(DAILY_CSV, index=False)

    summary_df = (
        daily_df
        .groupby(["snapshot_date", "provider"], as_index=False)
        .agg(
            model_count=("model_id", "nunique"),
            endpoint_count=("raw_provider_name", "count"),
            avg_cost_per_1m_tokens=("avg_cost_per_1m_tokens", "mean"),
            min_cost_per_1m_tokens=("avg_cost_per_1m_tokens", "min"),
            max_cost_per_1m_tokens=("avg_cost_per_1m_tokens", "max")
        )
        .sort_values(["snapshot_date", "avg_cost_per_1m_tokens"])
    )

    if SUMMARY_CSV.exists():
        old_summary_df = pd.read_csv(SUMMARY_CSV)

        old_summary_df = old_summary_df[
            old_summary_df["snapshot_date"] != SNAPSHOT_DATE
        ]

        final_summary_df = pd.concat(
            [old_summary_df, summary_df],
            ignore_index=True
        )
    else:
        final_summary_df = summary_df

    final_summary_df = final_summary_df.sort_values(
        ["snapshot_date", "avg_cost_per_1m_tokens"]
    )

    final_summary_df.to_csv(SUMMARY_CSV, index=False)

    return summary_df


def main():
    print(f"Starting OpenRouter cost tracking for {SNAPSHOT_DATE}")

    daily_df = fetch_daily_costs()

    update_historical_csv(daily_df)
    summary_df = update_daily_files(daily_df)

    print(f"Rows collected today: {len(daily_df)}")
    print(f"Historical CSV: {HISTORICAL_CSV}")
    print(f"Latest daily CSV: {DAILY_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")

    print("\nDaily provider summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
