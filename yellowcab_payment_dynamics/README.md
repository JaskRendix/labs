# NYC Yellow Taxi Trip Analysis

???

---

**What It Does:**

- Compares payment type distributions between 2019 and 2025.
- Analyzes average tip amounts by payment method and year.
- Identifies neighborhoods with the most dramatic shifts.
- Visualizes changes in payment behavior across NYC zones.

---

**Files of Interest:**

- `compare_payment_location.py` – The script that performed the analysis.
- `output_change_in_proportions/` – Directory containing the generated plots and summaries.

---

**Data Source:**

Trip data from the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- **Yellow Taxi Trip Records – January 2019 (PARQUET)**
- **Yellow Taxi Trip Records – January 2025 (PARQUET)**
- **Taxi Zone Lookup Table (CSV)**

---

**Data Summary:**

### Payment Type Distribution by Year
| Year | Flex Fare | Credit Card | Cash | No Charge | Dispute |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019 | 0.37% | 71.41% | 27.74% | 0.37% | 0.11% |
| 2025 | 13.68% | 73.42% | 11.28% | 0.46% | 1.17% |

### Average Tip Amount
| Year | Flex Fare | Credit Card | Cash | No Charge | Dispute |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019 | $0.01 | $2.55 | $0.00 | $0.00 | $0.00 |
| 2025 | $0.54 | $4.11 | $0.00 | $0.01 | $0.01 |

---

**Payment Type Shifts by Category:**

Visualizing how each payment method evolved between 2019 and 2025 across NYC neighborhoods.

### Flex Fare
A quiet revolution. From fringe to a significant feature.
![Flex Fare Shift](output_change_in_proportions/change_payment_type_0_Flex_Fare.png)

---

### Credit Card
Still dominant, but not untouched. Some zones saw dramatic swings.
![Credit Card Shift](output_change_in_proportions/change_payment_type_1_Credit_Card.png)

---

### Cash
The decline of the tangible. A farewell to folded bills.
![Cash Shift](output_change_in_proportions/change_payment_type_2_Cash.png)

---

### No Charge
Free rides—or accounting ghosts? A curious uptick.
![No Charge Shift](output_change_in_proportions/change_payment_type_3_No_Charge.png)

---

### Dispute
When the meter runs and the argument begins.
![Dispute Shift](output_change_in_proportions/change_payment_type_4_Dispute.png)

---

## Why It Exists

???

---

## What's Next?

???

