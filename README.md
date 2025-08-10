# Summer_Fellowship

Eli Swagel's Project

# Summer_Fellowship

**Automated Play Segmentation of Continuous GPS Data from Collegiate Football Players**  
*Wu-Tsai Undergraduate Fellowship – University of Oregon*  
Author: Eli Swagel  

## Overview
This project automates the identification of football plays from continuous high-frequency (10Hz) GPS tracking data collected from collegiate athletes. Using **time-series clustering** and **pattern matching**, it builds play templates based on metabolic power and applies them to find similar plays across games.  

This tool assists sports scientists in segmenting and analyzing plays on a play-by-play basis, enabling better workload management and performance analysis.

## Methods
1. **Data Processing**
   - GPS data is preprocessed (coordinate adjustments, time rounding).
   - Video annotation files are parsed to extract play time ranges.

2. **Feature Extraction**
   - Metabolic power (`mp`) is calculated and averaged over time.
   - Play segments are converted into time series datasets.

3. **Clustering**
   - `TimeSeriesKMeans` from `tslearn` groups similar plays into templates.
   - The elbow method helps determine the optimal number of clusters.

4. **Pattern Matching**
   - Templates are matched back to games using the `stumpy` library to detect similar plays.
   - Matches are annotated with period names and player participation stats.

## Repository Contents
- **`SummerFellowship.py`** – Main pipeline for data loading, clustering, and pattern matching.
- **`WuTsai_Fellowship_Presentation.pdf`** – Project presentation summarizing methodology and findings.

## Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn tslearn stumpy tqdm
```

## Usage
1. Place GPS (`.csv`) and video annotation files in the expected folder structure:

data/
├── 10hz/
├── catapult_activity_list/
└── video/

2. Update file paths inside `SummerFellowship.py` if needed.  
3. Run the script:

```bash
python SummerFellowship.py
```

Results
Generates metabolic power templates for different play types.
Produces plots for cluster centers and individual time series.
Outputs DataFrames of detected plays with:
Start and end times
Maximum metabolic power
Number of players on/off the field
