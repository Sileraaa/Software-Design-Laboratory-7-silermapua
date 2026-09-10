# Strava Running Data — Analysis Menu

## A. Purpose
This dataset (`Strava Running Data.xlsx`) contains one individual's activity history exported from Strava — 105 logged activities (mostly runs, with a few workouts/walks/a hike) spanning March 2022 to December 2023, with fields like distance, moving time, elevation gain, average/max speed, and location per activity. The script uses **pandas** to clean and organize the data, **scikit-learn** to fit a distance-vs-pace regression, and **matplotlib** to visualize it — the goal is to help this person interpret their own training volume, pace trend, consistency, and effort over time.

## B. Requirements
- Python 3
- Packages: `pandas`, `matplotlib`, `scikit-learn`, `openpyxl` (needed for pandas to read `.xlsx` files)
- Install with:
  ```
  python -m pip install pandas matplotlib scikit-learn openpyxl
  ```
- The file `Strava Running Data.xlsx` must be in the **same folder** as the script.

## C. How to Use
1. Run the script: `python strava_menu.py`
2. A bordered text menu appears with options 1–8.
3. Type a number and press Enter:
   - **1** – prints dataset metadata (shape, columns, date range, activity counts, missing values)
   - **2–7** – opens one matplotlib chart window (weekly volume, pace trend, distance vs. pace, day/hour heatmap, elevation vs. distance, distance distribution)
   - **8** – quits the program
4. Close the chart window to return to the menu and pick another option.

## D. Limitations
- Only **one chart can be viewed at a time** — the script pauses at `plt.show()` until you close that window before the menu reappears.
- No files are saved; charts exist only as live matplotlib windows during that session.
- No input validation beyond checking the number is 1–8 (e.g., typing letters just reprints the menu).
- Requires a local Python environment with a display (won't render chart windows in a headless/remote terminal).
- Analysis is scoped to this one person's data — not built for multi-user comparison.