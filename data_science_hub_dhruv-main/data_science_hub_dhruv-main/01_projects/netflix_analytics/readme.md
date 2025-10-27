# Netflix Content Analysis: Data-Driven Insights

A comprehensive analysis of Netflix's content catalog using Python, Pandas, and Machine Learning to uncover trends, patterns, and build predictive models.

---

## Project Overview

This project analyzes **8,793 Netflix titles** (movies and TV shows) to extract meaningful insights about content distribution, growth patterns, and viewing trends. The analysis includes extensive data cleaning, exploratory analysis, professional visualizations, and a machine learning model to predict content type.

### Key Objectives
- Clean and prepare real-world messy data
- Explore content distribution patterns and trends
- Visualize insights through professional charts
- Build a predictive ML model for content classification

---

## Key Findings

### Content Distribution
- **69.1% Movies** vs **30.9% TV Shows** on Netflix
- Most common ratings: **TV-MA** (3,205 titles) and **TV-14** (2,157 titles)
- Top content producer: **United States** (dominates the catalog)

### Growth Trends
- Netflix experienced **explosive growth from 2016-2019**
- **Peak year: 2019** with 2,016 titles added
- **Strategic shift after 2019**: Content additions declined (focus on quality over quantity)

### Content Characteristics
- Average movie duration: **~99 minutes**
- Content spans from **1925 to 2021** (nearly a century of entertainment)
- Most content released in the **2015-2020** period

### Machine Learning Model
- **95.34% accuracy** in predicting Movie vs TV Show
- No overfitting detected (training: 96.25%, test: 95.34%)
- Most important features: Release year, Rating, Year added

---

## Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **WordCloud** - Text visualization
- **Jupyter Notebook** - Interactive development

---

## Project Structure
netflix-analysis/
├── data/
│   └── netflix_titles.csv
├── notebooks/
│   └── netflix_analysis.ipynb
├── visualizations/
│   ├── viz_01_content_type_distribution.png
│   ├── viz_02_top_ratings.png
│   ├── viz_03_content_growth_timeline.png
│   ├── viz_04_top_countries.png
│   ├── viz_05_rating_by_type.png
│   ├── viz_06_release_year_distribution.png
│   ├── viz_07_content_type_timeline.png
│   ├── viz_08_rating_year_heatmap.png
│   ├── viz_09_movie_duration.png
│   ├── viz_10_title_wordcloud.png
│   └── viz_11_ml_model_performance.png
└── README.md
```

---

## Visualizations

### 1. Content Type Distribution
![Content Type](visualizations/viz_01_content_type_distribution.png)
*Netflix's catalog is dominated by movies (69.1%)*

### 2. Netflix Growth Timeline
![Growth Timeline](visualizations/viz_03_content_growth_timeline.png)
*Explosive growth 2016-2019, followed by strategic decline*

### 3. Rating Distribution Heatmap
![Heatmap](visualizations/viz_08_rating_year_heatmap.png)
*Content ratings distribution over time*

### 4. Word Cloud - Popular Title Words
![Word Cloud](visualizations/viz_10_title_wordcloud.png)
*Most common words appearing in Netflix titles*

### 5. ML Model Performance
![ML Performance](visualizations/viz_11_ml_model_performance.png)
*95.34% accuracy in content type prediction*

---

## Data Cleaning Process

The dataset required significant cleaning:
- **Dropped 14 rows** with missing critical data (rating, duration, date_added)
- **Filled 2,634 missing directors** with 'Unknown Director'
- **Filled 825 missing cast** entries with 'Cast Not Available'
- **Filled 831 missing countries** with 'Unknown Country'
- **Final clean dataset: 8,793 entries** (99.84% retention rate)

---

## Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Classifier
- **Features Used**:
  - Release year
  - Year added to Netflix
  - Content rating (encoded)
  - Director availability
  - US content flag
  - Title length

### Performance Metrics
- **Accuracy**: 95.34%
- **Precision**: 96% (Movies), 93% (TV Shows)
- **Recall**: 97% (Movies), 90% (TV Shows)
- **F1-Score**: 97% (Movies), 92% (TV Shows)

### Model Validation
- **No overfitting detected**
- Training accuracy (96.25%) is very close to test accuracy (95.34%)
- Model generalizes well to unseen data

---

## Key Insights

1. **Netflix's Strategic Shift**: After 2019, Netflix reduced content additions, likely focusing on original quality content over licensed quantity

2. **US Dominance**: The United States produces significantly more content than any other country on the platform

3. **Mature Content Focus**: TV-MA and TV-14 ratings dominate, indicating Netflix's focus on adult audiences

4. **Recent Content Preference**: Most content is from 2015-2020, showing Netflix's focus on contemporary entertainment

5. **Predictable Patterns**: 95% accuracy in ML model shows Movies and TV Shows have distinctly different characteristics

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/netflix-analysis.git
cd netflix-analysis
```

2. **Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

3. **Run the Jupyter notebook**
```bash
jupyter notebook notebooks/netflix_analysis.ipynb
```

---

## Dataset

**Source**: Netflix Movies and TV Shows dataset  
**Size**: 8,807 entries × 12 features  
**Features**: show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description

---

## Learning Outcomes

Through this project, I demonstrated proficiency in:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Data visualization best practices
- Feature engineering for ML
- Model training and evaluation
- Overfitting detection and validation
- Professional documentation

---

## 📬 Contact

**Dhruv Sharma**  
- LinkedIn: www.linkedin.com/in/dhruv-sharmaokstate
- GitHub: github.com/madwolfslayer/data_science_hub_dhruv
- Email: dhruvsharma14@gmail.com

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- Dataset: Kaggle Netflix Movies and TV Shows Dataset
- Inspired by real-world data analysis challenges
- Built as part of a continuous learning journey in Data Science

---

**If you found this project helpful, please give it a star!**
