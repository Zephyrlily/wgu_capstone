from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import tldextract
from urllib.parse import urlparse
import argparse, csv, json, os, re, sys, time, platform, io
from typing import List, Tuple
from url_normalize import url_normalize
from collections import Counter
from math import log2
from joblib import Parallel, delayed


# Constants
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)
IP_RE = re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)")
SCHEME_RE = re.compile(r"^https?://|^http://", re.I)
SUSPICIOUS_TLDS = {
    "li", "es", "sbs", "dev", "cfd", "ru", "pub", "so",
    "app", "icu", "re", "ua", "online", "link", "top",
    "ly", "site", "ro", "click", "ar"
}
BRAND_NAMES = {
    "microsoft", "apple", "google", "linkedin", "alibaba",
    "whatsapp", "amazon", "twitter", "facebook", "adobe",
    "bestbuy", "paypal", "bankofamerica", "wellsfargo"
}
SUSPICIOUS_ACTIONS = {
    "login", "verify", "account", "password", "secure",
    "confirm", "payment", "reset"
}
VERSION = 3

def normalize_url(url: str) -> str:
    u = (url or "").strip().replace("\\", "/")
    try:
        return url_normalize(u)
    except Exception:
        return u

# Gets the host and path from a link. The website's home page of sorts and where it leads.
def get_host_and_path(url: str) -> Tuple[str, str]:
    try:
        parsed_url = urlparse(normalize_url(url))
    except ValueError:
        return "", ""
    host = (parsed_url.hostname or "").lower()
    path = (parsed_url.path or "").lower()
    return host, path

def get_entropy(segment: str) -> float:
    if not segment:
        return 0.0
    counts = Counter(segment).values()
    total = sum(counts)
    ent = 0.0
    for c in counts:
        p = c / total
        ent -= p * log2(p)
    return ent

# Splits up the host into multiple parts
def get_tld_parts(host: str) -> Tuple[str, str, str]:
    if not host:
        return "", "", ""
    ext = EXTRACTOR(host)
    return ext.subdomain or "", ext.domain or "", ext.suffix or ""

def evaluate_url_features(url: str):
    url_text, pu = parse_url(url)
    host, path = get_host_and_path_from_parsed(pu)
    subdomain, domain, suffix = get_tld_parts(host)
    sub_labels = subdomain.split(".") if subdomain else []
    hp = (host + path)

    features = {
        "url_length": len(url_text),
        "digit_count": sum(c.isdigit() for c in url_text),
        "symbol_count": sum(not c.isalnum() for c in url_text),
        "dash_count": url_text.count("-"),
        "percent_count": url_text.count("%"),
        "contains_at": "@" in url_text,
        "has_ip_host": bool(IP_RE.match(url_text)),
        "host_entropy": get_entropy(host),
        "path_entropy": get_entropy(path),
        "suspicious_tld": int((suffix or "") in SUSPICIOUS_TLDS),
        "subdomains": len(sub_labels),
        "domain_length": len(domain or ""),
        "host_length": len(host),
        "path_length": len(path),
        "has_brand_token": int(any(tok in hp for tok in BRAND_NAMES)),
        "has_phish_token": int(any(tok in hp for tok in SUSPICIOUS_ACTIONS)),
    }
    return features

def parse_url(url: str):
    raw = (url or "").strip().replace("\\", "/")
    try:
        normal = url_normalize(raw)
    except Exception:
        normal = raw
    pu = urlparse(normal)
    return normal, pu

def get_host_and_path_from_parsed(pu):
    host = (pu.hostname or "").lower()
    path = (pu.path or "").lower()
    return host, path

# Read all the urls and values from the csv (url, is a phish label). This is used for training.
# 1 = phish, 0 = not phish
def read_csv_data(path: str) -> Tuple[List[str], List[int]]:
    urls, labels = [], []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "url" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} columns are incorrect. It should be url,label")
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))
    return urls, labels

# Read all the urls, but NOT the values. This is used for predicting.
def read_csv_urls(path: str) -> List[str]:
    urls = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "url" not in reader.fieldnames:
            raise ValueError(f"{path} should have a url column")
        for row in reader:
            urls.append(row["url"])
    return urls

def make_features(urls):
    return (Parallel(
        n_jobs=-1,
        prefer="threads",
        verbose=10
    )
    (delayed(evaluate_url_features)(u) for u in urls))

def train_model(urls, labels):
    features = make_features(urls)
    vector = DictVectorizer(sparse=True)
    data = vector.fit_transform(features)

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=400,
        verbose=2,
    )
    classifier.fit(data, labels)
    return {"vector": vector, "clf": classifier}

def predict_probability(bundle, urls):
    data = bundle["vector"].transform(make_features(urls))
    phish_index = list(bundle["clf"].classes_).index(1)
    return bundle["clf"].predict_proba(data)[:, phish_index].tolist()

# Save training data
def save_bundle(bundle, path: str):
    data = {
        "created": time.strftime("%Y-%m-%dT%H:%M", time.gmtime()),
        "python": platform.python_version(),
        "program_version": VERSION,
    }
    obj = {"data": data, "vector": bundle["vector"], "clf": bundle["clf"]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(obj, path, compress=3)

# Load data
def load_bundle(path: str):
    obj = joblib.load(path)
    version = obj.get("data", {}).get("program_version")
    if version > VERSION:
        print(f"WARNING: Loaded version is greater than program's. BE CAREFUL, it should be the same.", file=sys.stderr)
    elif version < VERSION:
        print(f"WARNING: Loaded version is lower than program's. BE CAREFUL, it should be the same.", file=sys.stderr)
    return obj

def train(args):
    urls, labels = read_csv_data(args.csv)
    bundle = train_model(urls, labels)
    save_bundle(bundle, args.model)
    print(json.dumps({"Model": args.model}))

def predict(args):
    bundle = load_bundle(args.model)
    urls = [args.url] if args.url else read_csv_urls(args.csv)
    probabilities = predict_probability(bundle, urls)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="")
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["url", "score", "prediction"])
    for u, p in zip(urls, probabilities):
        w.writerow([u, f"{p:.2f}", int(p >= args.threshold)])

def evaluate(args):
    bundle = load_bundle(args.model)
    urls, labels = read_csv_data(args.csv)
    probabilities = predict_probability(bundle, urls)

    threshold = float(args.threshold)
    predictions = [int(p >= threshold) for p in probabilities]
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)

    auc_str = f"{auc:.3f}"
    lines_to_print = [f"Threshold: {threshold}", f"Accuracy: {accuracy:.3f}", f"ROC AUC: {auc_str}"]
    print("\n".join(lines_to_print))


def command_handler():
    p = argparse.ArgumentParser(prog="phish_detector")
    sub = p.add_subparsers(required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--model", default="models/phish_detector.joblib")
    pt.set_defaults(func=train)

    # predict
    pp = sub.add_parser("predict")
    g = pp.add_mutually_exclusive_group(required=True)
    g.add_argument("--url")
    g.add_argument("--csv")
    pp.add_argument("--model", default="models/phish_detector.joblib")
    pp.add_argument("--threshold", type=float, default=0.55)
    pp.set_defaults(func=predict)

    # evaluate
    pe = sub.add_parser("evaluate")
    pe.add_argument("--csv", required=True)
    pe.add_argument("--model", default="models/phish_detector.joblib")
    pe.add_argument("--threshold", type=float, default=0.55)
    pe.set_defaults(func=evaluate)

    return p

def main():
    handler = command_handler()
    args = handler.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
