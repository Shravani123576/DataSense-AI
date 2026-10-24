import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from models import (
    DatasetMetadata,
    ColumnStat,
    TopCategory,
    CorrelationPair,
    HealthScoreBreakdown,
    ProfileReport
)



def try_clean_numeric_series(series: pd.Series) -> Optional[pd.Series]:
    """
    Attempt to coerce string columns that represent numbers (currency like '$1,200', '₹500', 
    percentages '15%', ratings '4.2/5', or numbers with commas '10,000') into float numbers.
    Does NOT coerce text categories that contain letters (e.g. 'Category_1', 'Batch_A').
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    # If it is an object/string series, check if it can be safely coerced
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return None

    sample = non_null.head(40)
    # If the strings contain letters (other than /5), it is an authentic text/categorical column
    sample_without_rating = sample.str.replace(r"[/]\s*5(\.0)?$", "", regex=True)
    letters_count = sample_without_rating.str.contains(r"[a-zA-Z]", regex=True).sum()
    if letters_count / len(sample) > 0.2:
        return None

    # Strip trailing '/5', currency symbols ($€£₹¥), %, commas, and spaces
    cleaned_sample = (
        sample_without_rating.str.replace(r"[$€£₹¥%,\s]", "", regex=True)
    )

    numeric_sample = pd.to_numeric(cleaned_sample, errors="coerce")
    valid_ratio = numeric_sample.notna().sum() / len(sample)
    
    # If at least 75% of non-empty sample strings parse as valid numbers, coerce the entire series
    if valid_ratio >= 0.75:
        cleaned_all = (
            series.astype(str).str.strip()
            .str.replace(r"[/]\s*5(\.0)?$", "", regex=True)
            .str.replace(r"[$€£₹¥%,\s]", "", regex=True)
        )
        return pd.to_numeric(cleaned_all, errors="coerce")
    
    return None


def clean_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a sanitized, properly typed DataFrame for analysis.
    Coerces disguised numeric columns and date columns into real analytical types.
    """
    clean_df = df.copy()
    for col in clean_df.columns:
        # Check if disguised numeric
        if not pd.api.types.is_numeric_dtype(clean_df[col]):
            num_series = try_clean_numeric_series(clean_df[col])
            if num_series is not None:
                clean_df[col] = num_series
                continue
                
            # Check if disguised datetime
            non_null_str = clean_df[col].dropna().astype(str)
            if len(non_null_str) > 0:
                sample = non_null_str.head(20)
                if any("-" in s or "/" in s for s in sample):
                    try:
                        parsed_dates = pd.to_datetime(clean_df[col], errors="coerce", format="mixed")
                        if parsed_dates.notna().sum() / len(clean_df[col].dropna()) >= 0.75:
                            clean_df[col] = parsed_dates
                    except Exception:
                        pass
                        
    return clean_df


def infer_column_type(series: pd.Series) -> str:
    """Infer the semantic type of a pandas Series."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # Check if numeric column is actually a boolean or binary flag
        non_null = series.dropna()
        if len(non_null) > 0 and set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
            return "boolean"
        return "numeric"
    
    # Try parsing string to datetime if possible
    non_null_str = series.dropna().astype(str)
    if len(non_null_str) > 0:
        sample = non_null_str.iloc[:10]
        try:
            if any("-" in s or "/" in s for s in sample):
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().sum() >= len(sample) * 0.7:
                    return "datetime"
        except (ValueError, TypeError):
            pass

    # High cardinality vs categorical
    unique_count = series.nunique(dropna=True)
    total_count = len(series.dropna())
    if total_count > 0 and (unique_count / total_count > 0.8) and unique_count > 25:
        return "text"
    return "categorical"


def sanitize_for_json(val: Any) -> Any:
    """Safely convert numpy/pandas types and NaNs/Infs to clean JSON serializable values."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 3)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    return str(val)


def calculate_data_quality_score(
    df: pd.DataFrame,
    duplicate_rows: int,
    missing_cells: int,
    total_cells: int,
    outlier_counts: List[int]
) -> HealthScoreBreakdown:
    total_rows = len(df)
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
    dup_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0
    
    # Completeness Score: 0 to 100
    completeness_score = max(0, min(100, int(100 - (missing_pct * 2.5))))
    
    # Uniqueness Score: 0 to 100
    uniqueness_score = max(0, min(100, int(100 - (dup_pct * 10))))
    
    # Outlier Score: 0 to 100
    num_numeric = len(outlier_counts)
    total_outliers = sum(outlier_counts)
    if num_numeric > 0 and total_rows > 0:
        outlier_pct = (total_outliers / (num_numeric * total_rows)) * 100
        outlier_score = max(0, min(100, int(100 - (outlier_pct * 3.5))))
    else:
        outlier_score = 100

    # Consistency Score: column types stability
    consistency_score = 95
    if missing_pct > 15:
        consistency_score -= 15
    if dup_pct > 5:
        consistency_score -= 10
    consistency_score = max(30, min(100, consistency_score))

    # Overall Score
    overall_score = int(
        0.35 * completeness_score +
        0.25 * uniqueness_score +
        0.20 * outlier_score +
        0.20 * consistency_score
    )
    overall_score = max(10, min(100, overall_score))

    # Grade
    if overall_score >= 92:
        grade = "A+"
    elif overall_score >= 84:
        grade = "A"
    elif overall_score >= 74:
        grade = "B"
    elif overall_score >= 62:
        grade = "C"
    else:
        grade = "Needs Attention"

    # Badges
    badges = []
    if missing_cells == 0:
        badges.append("100% Complete (No Missing Values)")
    elif missing_pct < 3:
        badges.append("Highly Complete (>97% filled)")
        
    if duplicate_rows == 0:
        badges.append("Zero Duplicate Rows")
        
    if outlier_score >= 90:
        badges.append("Clean Distributions (Low Outlier Rate)")
        
    if overall_score >= 85:
        badges.append("Production-Ready Data Health")

    # Recommendations
    recommendations = []
    if duplicate_rows > 0:
        recommendations.append(
            f"Found {duplicate_rows} exact duplicate row(s). Deduplicate to prevent inflated aggregations."
        )
    if missing_cells > 0:
        recommendations.append(
            f"Detected {missing_cells} missing cell(s) ({missing_pct:.1f}% of dataset). Impute with mean/median or drop depending on importance."
        )
    if total_outliers > 0:
        recommendations.append(
            f"Identified {total_outliers} statistical outlier value(s) across numeric columns using the 1.5x IQR rule. Verify if these represent valid extremes or recording errors."
        )
    if len(recommendations) == 0:
        recommendations.append(
            "Dataset is in pristine analytical shape with no critical quality anomalies detected."
        )

    return HealthScoreBreakdown(
        overall_score=overall_score,
        completeness_score=completeness_score,
        uniqueness_score=uniqueness_score,
        outlier_score=outlier_score,
        consistency_score=consistency_score,
        grade=grade,
        badges=badges,
        recommendations=recommendations
    )


def compute_column_stat(series: pd.Series, col_name: str) -> Tuple[ColumnStat, int]:
    """Compute detailed statistics for a single column. Returns (ColumnStat, outlier_count)."""
    total_rows = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    
    unique_count = int(series.nunique(dropna=True))
    unique_pct = round((unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    
    inferred_type = infer_column_type(series)
    
    # Sample clean values
    clean_sample = [
        sanitize_for_json(v)
        for v in series.dropna().head(5).tolist()
    ]
    
    stat = ColumnStat(
        column=col_name,
        dtype=str(series.dtype),
        inferred_type=inferred_type,
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
        unique_percentage=unique_pct,
        sample_values=clean_sample
    )
    
    outlier_count = 0
    
    if inferred_type == "numeric":
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) > 0:
            stat.min = sanitize_for_json(numeric_series.min())
            stat.max = sanitize_for_json(numeric_series.max())
            stat.mean = sanitize_for_json(numeric_series.mean())
            stat.median = sanitize_for_json(numeric_series.median())
            stat.std = sanitize_for_json(numeric_series.std()) if len(numeric_series) > 1 else 0.0
            
            q25 = numeric_series.quantile(0.25)
            q75 = numeric_series.quantile(0.75)
            iqr = q75 - q25
            
            stat.q25 = sanitize_for_json(q25)
            stat.q75 = sanitize_for_json(q75)
            stat.iqr = sanitize_for_json(iqr)
            
            if len(numeric_series) > 2 and numeric_series.std() > 0:
                stat.skewness = sanitize_for_json(numeric_series.skew())
            else:
                stat.skewness = 0.0
                
            # Outlier detection using 1.5 * IQR rule
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
                outlier_count = len(outliers)
                stat.outlier_count = outlier_count
                stat.outlier_percentage = round((outlier_count / len(numeric_series)) * 100, 2)
            else:
                stat.outlier_count = 0
                stat.outlier_percentage = 0.0
                
    elif inferred_type in ["categorical", "boolean", "text"]:
        # Compute top value counts
        val_counts = series.value_counts(dropna=True).head(6)
        total_valid = series.count()
        top_cats = []
        for val, count in val_counts.items():
            pct = round((count / total_valid * 100), 1) if total_valid > 0 else 0.0
            top_cats.append(TopCategory(
                value=str(val),
                count=int(count),
                percentage=pct
            ))
        stat.top_categories = top_cats
        
    return stat, outlier_count


def compute_correlations(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], List[CorrelationPair]]:
    """Compute Pearson correlation matrix and extract top correlated pairs with 100k+ row optimization."""
    numeric_df = df.select_dtypes(include=[np.number])
    valid_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() > 1]
    
    if len(valid_cols) < 2:
        return {}, []
        
    # High-performance sampling for large datasets (e.g. 100k+ rows) to ensure sub-second latency
    if len(numeric_df) > 25000:
        sample_df = numeric_df[valid_cols].sample(20000, random_state=42)
        corr_df = sample_df.corr()
    else:
        corr_df = numeric_df[valid_cols].corr()
    
    matrix_dict: Dict[str, Dict[str, float]] = {}
    pairs_list: List[CorrelationPair] = []
    seen = set()
    
    for c1 in valid_cols:
        matrix_dict[c1] = {}
        for c2 in valid_cols:
            val = corr_df.loc[c1, c2]
            clean_val = 0.0 if (pd.isna(val) or math.isnan(val)) else round(float(val), 3)
            matrix_dict[c1][c2] = clean_val
            
            if c1 != c2:
                pair_key = tuple(sorted([c1, c2]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    if abs(clean_val) >= 0.15:
                        pairs_list.append(CorrelationPair(
                            col1=pair_key[0],
                            col2=pair_key[1],
                            correlation=clean_val
                        ))
                        
    # Sort pairs by absolute correlation descending
    pairs_list.sort(key=lambda x: abs(x.correlation), reverse=True)
    return matrix_dict, pairs_list[:10]


def extract_restaurant_insights(df: pd.DataFrame, dataset_id: str = "") -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Detect Zomato/restaurant domain features and extract specialized culinary & operational insights."""
    col_map = {str(c).lower().strip(): str(c) for c in df.columns}
    
    # Identify key restaurant attributes
    rating_col = None
    for k in ["rating", "rate", "stars", "overall_rating"]:
        if k in col_map:
            rating_col = col_map[k]
            break
            
    cuisine_col = None
    for k in ["cuisine", "cuisines", "food_type"]:
        if k in col_map:
            cuisine_col = col_map[k]
            break
            
    cost_col = None
    for k in ["cost_for_two", "approx_cost(for two people)", "approx_cost", "price_for_two", "average_cost"]:
        if k in col_map:
            cost_col = col_map[k]
            break
            
    name_col = None
    for k in ["restaurant_name", "restaurant", "place_name"]:
        if k in col_map:
            name_col = col_map[k]
            break
    if not name_col and "name" in col_map and ("cuisine" in col_map or "cost_for_two" in col_map):
        name_col = col_map["name"]

    loc_col = None
    for k in ["location", "locality", "city", "area", "neighborhood"]:
        if k in col_map:
            loc_col = col_map[k]
            break

    # Strict check: Must be explicit Zomato sample OR have rating AND cuisine AND (cost_for_two or restaurant_name)
    is_explicit_sample = "zomato" in dataset_id.lower()
    has_strong_markers = bool(rating_col and cuisine_col and (cost_col or name_col))
    
    if not (is_explicit_sample or has_strong_markers):
        return False, None

    # Parse numeric ratings (handles "4.2/5", "4.2", etc.)
    ratings_numeric = pd.to_numeric(
        df[rating_col].astype(str).str.extract(r"([\d\.]+)")[0],
        errors="coerce"
    )
    
    # Parse numeric cost (handles "800", "1,200", etc.)
    cost_numeric = None
    if cost_col:
        cost_numeric = pd.to_numeric(
            df[cost_col].astype(str).str.replace(",", "").str.extract(r"(\d+)")[0],
            errors="coerce"
        )

    total_places = len(df)
    avg_rating = round(float(ratings_numeric.dropna().mean()), 2) if len(ratings_numeric.dropna()) > 0 else 0.0

    # Top cuisines by frequency & quality
    top_cuisines = []
    if cuisine_col:
        cuisine_series = df[cuisine_col].dropna().astype(str)
        top_cats = cuisine_series.value_counts().head(8)
        for c_val, cnt in top_cats.items():
            mask = df[cuisine_col].astype(str).str.contains(re.escape(str(c_val)), na=False)
            sub_ratings = ratings_numeric[mask].dropna()
            sub_cost = cost_numeric[mask].dropna() if cost_numeric is not None else None
            top_cuisines.append({
                "cuisine": str(c_val),
                "count": int(cnt),
                "avg_rating": round(float(sub_ratings.mean()), 2) if len(sub_ratings) > 0 else None,
                "avg_cost": int(round(float(sub_cost.mean()))) if sub_cost is not None and len(sub_cost) > 0 else None
            })

    # Top dining locations
    top_locations = []
    if loc_col:
        loc_counts = df[loc_col].dropna().value_counts().head(8)
        for l_val, cnt in loc_counts.items():
            mask = df[loc_col] == l_val
            sub_ratings = ratings_numeric[mask].dropna()
            sub_cost = cost_numeric[mask].dropna() if cost_numeric is not None else None
            top_locations.append({
                "location": str(l_val),
                "count": int(cnt),
                "avg_rating": round(float(sub_ratings.mean()), 2) if len(sub_ratings) > 0 else None,
                "avg_cost": int(round(float(sub_cost.mean()))) if sub_cost is not None and len(sub_cost) > 0 else None
            })

    # Top rated restaurants
    top_rated_restaurants = []
    if name_col:
        temp_df = df.copy()
        temp_df["_num_rating"] = ratings_numeric
        if cost_numeric is not None:
            temp_df["_num_cost"] = cost_numeric
            
        valid_df = temp_df.dropna(subset=["_num_rating"])
        votes_col = col_map.get("votes")
        if votes_col and pd.api.types.is_numeric_dtype(valid_df[votes_col]):
            sorted_df = valid_df.sort_values(by=["_num_rating", votes_col], ascending=[False, False])
        else:
            sorted_df = valid_df.sort_values(by="_num_rating", ascending=False)

        for _, r in sorted_df.head(10).iterrows():
            top_rated_restaurants.append({
                "name": str(r[name_col]),
                "rating": round(float(r["_num_rating"]), 1),
                "location": str(r[loc_col]) if loc_col else "N/A",
                "cuisine": str(r[cuisine_col]) if cuisine_col else "N/A",
                "cost": int(r["_num_cost"]) if cost_numeric is not None and pd.notna(r.get("_num_cost")) else None,
                "votes": int(r[votes_col]) if votes_col and pd.notna(r.get(votes_col)) else None
            })

    # Cost vs rating correlation
    cost_vs_rating_corr = None
    if cost_numeric is not None and len(ratings_numeric.dropna()) > 0 and len(cost_numeric.dropna()) > 0:
        c_df = pd.DataFrame({"rating": ratings_numeric, "cost": cost_numeric}).dropna()
        if len(c_df) > 5:
            cost_vs_rating_corr = round(float(c_df["rating"].corr(c_df["cost"])), 3)

    # Online order comparison
    online_order_comp = None
    online_col = col_map.get("online_order") or col_map.get("online_ordering")
    if online_col:
        online_yes = ratings_numeric[df[online_col].astype(str).str.lower().isin(["yes", "true", "1"])].dropna()
        online_no = ratings_numeric[df[online_col].astype(str).str.lower().isin(["no", "false", "0"])].dropna()
        online_order_comp = {
            "yes_count": len(online_yes),
            "yes_avg_rating": round(float(online_yes.mean()), 2) if len(online_yes) > 0 else None,
            "no_count": len(online_no),
            "no_avg_rating": round(float(online_no.mean()), 2) if len(online_no) > 0 else None,
        }

    # Table booking comparison
    booking_comp = None
    book_col = col_map.get("table_booking") or col_map.get("book_table")
    if book_col:
        book_yes = ratings_numeric[df[book_col].astype(str).str.lower().isin(["yes", "true", "1"])].dropna()
        book_no = ratings_numeric[df[book_col].astype(str).str.lower().isin(["no", "false", "0"])].dropna()
        booking_comp = {
            "yes_count": len(book_yes),
            "yes_avg_rating": round(float(book_yes.mean()), 2) if len(book_yes) > 0 else None,
            "no_count": len(book_no),
            "no_avg_rating": round(float(book_no.mean()), 2) if len(book_no) > 0 else None,
        }

    # Review highlight snippets
    review_col = None
    for k in ["review_highlight", "reviews_list", "reviews", "review", "review_snippets"]:
        if k in col_map:
            review_col = col_map[k]
            break
            
    sample_reviews = []
    if review_col and name_col:
        sample_df = df.dropna(subset=[review_col, name_col]).head(6)
        for _, r in sample_df.iterrows():
            sample_reviews.append({
                "restaurant": str(r[name_col]),
                "highlight": str(r[review_col])[:180],
                "rating": round(float(ratings_numeric[r.name]), 1) if pd.notna(ratings_numeric[r.name]) else None
            })

    insights = {
        "total_restaurants": total_places,
        "avg_rating": avg_rating,
        "rating_column": rating_col,
        "cost_column": cost_col,
        "cuisine_column": cuisine_col,
        "location_column": loc_col,
        "top_cuisines": top_cuisines,
        "top_locations": top_locations,
        "top_rated_restaurants": top_rated_restaurants,
        "cost_vs_rating_corr": cost_vs_rating_corr,
        "online_order_comp": online_order_comp,
        "booking_comp": booking_comp,
        "sample_reviews": sample_reviews
    }
    return True, insights


def generate_suggested_questions(df: pd.DataFrame, columns_stats: List[ColumnStat], is_restaurant: bool = False) -> List[str]:
    """Generate dynamic beginner-friendly natural language questions based on the dataset schema."""
    if is_restaurant:
        return [
            "Which cuisines have the highest average rating?",
            "What is the correlation between cost for two and restaurant rating?",
            "Which locations have the highest concentration of restaurants?",
            "Do restaurants with online ordering have better ratings?",
            "Show me the top 10 restaurants with the highest votes"
        ]

    numeric_cols = [s.column for s in columns_stats if s.inferred_type == "numeric"]
    cat_cols = [s.column for s in columns_stats if s.inferred_type in ["categorical", "boolean"]]
    
    questions = []
    if numeric_cols and cat_cols:
        questions.append(f"What is the average {numeric_cols[0]} by {cat_cols[0]}?")
    if len(cat_cols) > 0:
        questions.append(f"What is the breakdown of {cat_cols[0]}?")
    if len(numeric_cols) >= 2:
        questions.append(f"Is there a strong correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        questions.append(f"Which {cat_cols[0]} has the highest total {numeric_cols[0]}?")
    if len(numeric_cols) > 0:
        questions.append(f"Show me the distribution and summary statistics of {numeric_cols[0]}")
    questions.append("What are the main data quality issues or missing values in this dataset?")
    
    return questions[:5]


def profile_dataframe(
    df: pd.DataFrame,
    dataset_id: str,
    name: str,
    filename: str,
    preset_questions: List[str] = None
) -> ProfileReport:
    """Run full automated profiling on a DataFrame after clean type preparation."""
    # Coerce disguised numeric and date columns so calculations are 100% accurate
    df = clean_and_prepare_dataframe(df)

    total_rows = len(df)
    total_cols = len(df.columns)
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    
    metadata = DatasetMetadata(
        id=dataset_id,
        name=name,
        filename=filename,
        rows=total_rows,
        columns=total_cols,
        memory_kb=memory_kb,
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        missing_percentage=missing_pct,
        column_names=list(df.columns)
    )
    
    columns_stats: List[ColumnStat] = []
    outlier_counts: List[int] = []
    
    for col in df.columns:
        stat, outliers = compute_column_stat(df[col], str(col))
        columns_stats.append(stat)
        if stat.inferred_type == "numeric":
            outlier_counts.append(outliers)
            
    health = calculate_data_quality_score(
        df=df,
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        total_cells=total_cells,
        outlier_counts=outlier_counts
    )
    
    corr_matrix, top_corrs = compute_correlations(df)
    
    # Check for restaurant / Zomato domain features (only if explicit or strong markers)
    is_restaurant, restaurant_insights = extract_restaurant_insights(df, dataset_id=dataset_id)
    
    # Preview rows (convert NaNs to None)
    preview_df = df.head(15).copy()
    preview_rows = []
    for _, row in preview_df.iterrows():
        row_dict = {str(k): sanitize_for_json(v) for k, v in row.items()}
        preview_rows.append(row_dict)
        
    suggested = preset_questions if preset_questions else generate_suggested_questions(df, columns_stats, is_restaurant)
    
    return ProfileReport(
        metadata=metadata,
        columns=columns_stats,
        health=health,
        correlation_matrix=corr_matrix,
        top_correlations=top_corrs,
        preview_rows=preview_rows,
        suggested_questions=suggested,
        is_restaurant_dataset=is_restaurant,
        restaurant_insights=restaurant_insights
    )

