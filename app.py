from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import io, json, re
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_df(file_storage):
    name = file_storage.filename.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file_storage)
    elif name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_storage)
    raise ValueError("Unsupported file type — use CSV or Excel.")


# ─────────────────────────────────────────────────────────────────────────────
# PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def profile(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    total_cells = df.shape[0] * df.shape[1]

    stats = {}
    for col in num_cols:
        s = df[col].dropna()
        if len(s):
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        else:
            q1 = q3 = iqr = outliers = 0
        stats[col] = {
            'mean':     round(float(df[col].mean()), 4) if not df[col].isna().all() else None,
            'median':   round(float(df[col].median()), 4) if not df[col].isna().all() else None,
            'std':      round(float(df[col].std()), 4) if not df[col].isna().all() else None,
            'min':      round(float(df[col].min()), 4) if not df[col].isna().all() else None,
            'max':      round(float(df[col].max()), 4) if not df[col].isna().all() else None,
            'q1':       round(float(q1), 4),
            'q3':       round(float(q3), 4),
            'outliers': outliers,
        }

    missing_per_col   = df.isnull().sum().to_dict()
    missing_pct_col   = {k: round(100*v/len(df), 1) for k, v in missing_per_col.items()}

    # Infer potential date columns from object cols
    potential_dates = []
    for col in cat_cols:
        sample = df[col].dropna().head(20).astype(str)
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        if any(sample.str.match(p).mean() > 0.5 for p in date_patterns):
            potential_dates.append(col)

    return {
        'rows':            int(df.shape[0]),
        'cols':            int(df.shape[1]),
        'columns':         list(df.columns),
        'dtypes':          {c: str(df[c].dtype) for c in df.columns},
        'cardinality':     {c: int(df[c].nunique()) for c in df.columns},
        'missing':         {k: int(v) for k, v in missing_per_col.items()},
        'missing_pct':     missing_pct_col,
        'total_missing':   int(df.isnull().sum().sum()),
        'missing_pct_total': round(100 * df.isnull().sum().sum() / total_cells, 2) if total_cells else 0,
        'duplicates':      int(df.duplicated().sum()),
        'num_cols':        num_cols,
        'cat_cols':        cat_cols,
        'date_cols':       date_cols,
        'potential_dates': potential_dates,
        'stats':           stats,
        'sample':          df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient='records'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING MODES
# ─────────────────────────────────────────────────────────────────────────────

def clean_basic(df: pd.DataFrame, steps: list) -> tuple[pd.DataFrame, list]:
    """
    Basic cleaning: user picks individual steps.
    steps can include any of:
      duplicates, missing, invalid, formats, irrelevant,
      consistency, outliers, dtypes
    """
    log = []

    if 'duplicates' in steps:
        before = len(df)
        df = df.drop_duplicates()
        log.append({'step': 'Remove Duplicates', 'icon': '🗑️',
                    'detail': f'Removed {before - len(df)} duplicate row(s). {len(df)} rows remain.'})

    if 'missing' in steps:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.7]
        if high_miss:
            df = df.drop(columns=high_miss)
            log.append({'step': 'Missing Values', 'icon': '🔧',
                        'detail': f'Dropped {len(high_miss)} column(s) with >70% missing: {", ".join(high_miss)}'})
            num_cols = [c for c in num_cols if c not in high_miss]
            cat_cols = [c for c in cat_cols if c not in high_miss]
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown')
        total_filled = df.isnull().sum().sum()
        log.append({'step': 'Missing Values', 'icon': '🔧',
                    'detail': f'Filled missing values — median for numeric, mode for categorical. Remaining: {total_filled}'})

    if 'invalid' in steps:
        fixed = 0
        for c in df.select_dtypes(include=np.number).columns:
            # Replace inf/-inf with NaN then median
            inf_count = np.isinf(df[c]).sum()
            if inf_count:
                df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())
                fixed += int(inf_count)
        # Strip leading/trailing whitespace from strings
        str_cols = df.select_dtypes(include='object').columns
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace('nan', np.nan)
        log.append({'step': 'Fix Invalid Data', 'icon': '⚠️',
                    'detail': f'Replaced {fixed} infinite value(s) with median. Stripped whitespace from {len(str_cols)} text column(s).'})

    if 'formats' in steps:
        standardized = []
        for c in df.select_dtypes(include='object').columns:
            sample = df[c].dropna().head(30).astype(str)
            # Standardize emails → lowercase
            if sample.str.contains('@').mean() > 0.5:
                df[c] = df[c].str.lower().str.strip()
                standardized.append(f'{c} (email→lowercase)')
            # Standardize phone-like → digits only
            elif sample.str.replace(r'[\d\s\-\(\)\+]', '', regex=True).str.len().mean() < 1:
                df[c] = df[c].str.replace(r'[^\d]', '', regex=True)
                standardized.append(f'{c} (phone→digits)')
        log.append({'step': 'Standardize Formats', 'icon': '📐',
                    'detail': f'Standardized {len(standardized)} column(s): {", ".join(standardized) if standardized else "no auto-detectable formats found"}'})

    if 'irrelevant' in steps:
        dropped = []
        # Drop constant columns
        for c in df.columns:
            if df[c].nunique() <= 1:
                df = df.drop(columns=[c])
                dropped.append(f'{c} (constant)')
        # Drop columns that look like row IDs (monotonically increasing integers)
        for c in df.select_dtypes(include=np.number).columns:
            if df[c].is_monotonic_increasing and df[c].nunique() == len(df):
                df = df.drop(columns=[c])
                dropped.append(f'{c} (ID-like)')
        log.append({'step': 'Remove Irrelevant', 'icon': '✂️',
                    'detail': f'Removed {len(dropped)} column(s): {", ".join(dropped) if dropped else "none detected"}'})

    if 'consistency' in steps:
        issues = []
        # Check for negative values in likely-positive columns
        for c in df.select_dtypes(include=np.number).columns:
            if c.lower() in ('age', 'price', 'salary', 'amount', 'quantity', 'count', 'weight', 'height'):
                neg = (df[c] < 0).sum()
                if neg:
                    df[c] = df[c].abs()
                    issues.append(f'{c}: {neg} negative→abs()')
        log.append({'step': 'Consistency Check', 'icon': '🔗',
                    'detail': f'Fixed {len(issues)} consistency issue(s): {", ".join(issues) if issues else "no obvious issues detected"}'})

    if 'outliers' in steps:
        capped = []
        for c in df.select_dtypes(include=np.number).columns:
            q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            n = int(((df[c] < lo) | (df[c] > hi)).sum())
            if n:
                df[c] = df[c].clip(lo, hi)
                capped.append(f'{c} ({n})')
        log.append({'step': 'Outlier Treatment', 'icon': '📊',
                    'detail': f'Capped outliers (IQR method) in {len(capped)} column(s): {", ".join(capped) if capped else "no outliers found"}'})

    if 'dtypes' in steps:
        converted = []
        for c in df.select_dtypes(include='object').columns:
            # Try numeric
            converted_col = pd.to_numeric(df[c], errors='coerce')
            if converted_col.notna().mean() > 0.9:
                df[c] = converted_col
                converted.append(f'{c} → numeric')
                continue
            # Try datetime
            try:
                converted_col = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                if converted_col.notna().mean() > 0.9:
                    df[c] = converted_col
                    converted.append(f'{c} → datetime')
            except Exception:
                pass
        log.append({'step': 'Validate Data Types', 'icon': '🏷️',
                    'detail': f'Converted {len(converted)} column(s): {", ".join(converted) if converted else "all types already correct"}'})

    return df, log


def clean_ml(df: pd.DataFrame, options: dict) -> tuple[pd.DataFrame, list]:
    """Full ML pipeline: dedupe → missing → encode → scale."""
    log = []

    # Duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.append({'step': 'Remove Duplicates', 'icon': '🗑️',
                'detail': f'Removed {before - len(df)} duplicate row(s).'})

    # Drop high-missing columns
    high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.5]
    if high_miss:
        df = df.drop(columns=high_miss)
        log.append({'step': 'High-Missing Cols', 'icon': '🗑️',
                    'detail': f'Dropped {len(high_miss)} column(s) with >50% missing: {", ".join(high_miss)}'})

    # Missing values
    strategy = options.get('missing', 'smart')
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    target = options.get('target', '')

    if strategy == 'drop':
        before = len(df)
        df = df.dropna()
        log.append({'step': 'Missing Values', 'icon': '🔧', 'detail': f'Dropped {before - len(df)} rows with any missing value.'})
    elif strategy == 'knn':
        if num_cols:
            imp = KNNImputer(n_neighbors=5)
            df[num_cols] = imp.fit_transform(df[num_cols])
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown')
        log.append({'step': 'Missing Values', 'icon': '🔧', 'detail': 'KNN imputation (numeric) + mode (categorical).'})
    elif strategy == 'mean':
        for c in num_cols: df[c] = df[c].fillna(df[c].mean())
        for c in cat_cols: df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown')
        log.append({'step': 'Missing Values', 'icon': '🔧', 'detail': 'Filled — mean (numeric), mode (categorical).'})
    else:
        for c in num_cols: df[c] = df[c].fillna(df[c].median())
        for c in cat_cols: df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown')
        log.append({'step': 'Missing Values', 'icon': '🔧', 'detail': 'Filled — median (numeric), mode (categorical).'})

    # Inf values
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())

    # Encoding
    enc = options.get('encoding', 'auto')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target and target in cat_cols: cat_cols.remove(target)
    encoded = []
    for c in cat_cols:
        card = df[c].nunique()
        if enc == 'label' or (enc == 'auto' and card > 10):
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoded.append(f'{c}(label)')
        else:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
            encoded.append(f'{c}(one-hot,{card})')
    if target and target in df.columns and df[target].dtype == object:
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
        encoded.append(f'{target}[TARGET]')
    log.append({'step': 'Encoding', 'icon': '🔤',
                'detail': f'Encoded {len(encoded)} col(s): {", ".join(encoded) if encoded else "none"}'})

    # Scaling
    scale = options.get('scaling', 'standard')
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target and target in num_cols: num_cols.remove(target)
    if num_cols:
        scaler = StandardScaler() if scale == 'standard' else MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        label = 'StandardScaler z-score' if scale == 'standard' else 'MinMaxScaler 0–1'
        log.append({'step': 'Feature Scaling', 'icon': '📏',
                    'detail': f'Scaled {len(num_cols)} numeric feature(s) using {label}.'})

    return df, log


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        df = load_df(request.files['file'])
        return jsonify({'success': True, 'profile': profile(df)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/clean', methods=['POST'])
def clean():
    try:
        file    = request.files['file']
        options = json.loads(request.form.get('options', '{}'))
        mode    = options.get('mode', 'basic')

        df = load_df(file)
        before = profile(df)

        if mode == 'ml':
            cleaned, log = clean_ml(df.copy(), options)
        else:
            steps = options.get('steps', ['duplicates', 'missing'])
            cleaned, log = clean_basic(df.copy(), steps)

        after = profile(cleaned)

        buf = io.StringIO()
        cleaned.to_csv(buf, index=False)

        return jsonify({
            'success':        True,
            'log':            log,
            'before_profile': before,
            'after_profile':  after,
            'csv_data':       buf.getvalue(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print('\n⚡  SLAM is live → http://localhost:5000\n')
    app.run(debug=True, port=5000)
