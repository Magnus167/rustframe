use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::result::Result;
use std::str::FromStr;

// --- Core Enums ---

/// Represents the frequency at which calendar dates should be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateFreq {
    Daily,        // Every calendar day
    WeeklyMonday, // Every Monday
    WeeklyFriday, // Every Friday
    MonthStart,   // First calendar day of the month
    MonthEnd,     // Last calendar day of the month
    QuarterStart, // First calendar day of the quarter
    QuarterEnd,   // Last calendar day of the quarter
    YearStart,    // First calendar day of the year (Jan 1st)
    YearEnd,      // Last calendar day of the year (Dec 31st)
}

/// Indicates whether the first or last date in a periodic group (like month, quarter)
/// is selected for the frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    Start, // Indicates picking the first calendar date in a group's period.
    End,   // Indicates picking the last calendar day in a group's period.
}

impl DateFreq {
    /// Attempts to parse a frequency string into a `DateFreq` enum.
    ///
    /// This is a convenience wrapper around `from_str`.
    ///
    /// # Arguments
    ///
    /// * `freq` - The frequency string (e.g., "D", "W", "ME").
    ///
    /// # Errors
    ///
    /// Returns an error if the string does not match any known frequency.
    pub fn from_string(freq: String) -> Result<Self, Box<dyn Error>> {
        freq.parse()
    }

    /// Returns the canonical string representation of the frequency.
    ///
    /// This returns the primary code (e.g., "D", "W", "Y", "YE"), not the aliases.
    pub fn to_string(&self) -> String {
        let r = match self {
            DateFreq::Daily => "D",
            DateFreq::WeeklyMonday => "W",
            DateFreq::MonthStart => "M",
            DateFreq::QuarterStart => "Q",
            DateFreq::YearStart => "Y",
            DateFreq::MonthEnd => "ME",
            DateFreq::QuarterEnd => "QE",
            DateFreq::WeeklyFriday => "WF",
            DateFreq::YearEnd => "YE",
        };
        r.to_string()
    }

    /// Determines whether the frequency represents a start-of-period or end-of-period aggregation.
    pub fn agg_type(&self) -> AggregationType {
        match self {
            DateFreq::Daily
            | DateFreq::WeeklyMonday
            | DateFreq::MonthStart
            | DateFreq::QuarterStart
            | DateFreq::YearStart => AggregationType::Start,

            DateFreq::WeeklyFriday
            | DateFreq::MonthEnd
            | DateFreq::QuarterEnd
            | DateFreq::YearEnd => AggregationType::End,
        }
    }
}

// Implement FromStr for DateFreq to allow parsing directly using `parse()`
impl FromStr for DateFreq {
    type Err = Box<dyn Error>;

    /// Attempts to parse a frequency string slice into a `DateFreq` enum.
    ///
    /// Supports various frequency codes and common aliases.
    ///
    /// | Code | Alias   | Description              |
    /// |------|---------|--------------------------|
    /// | D    |         | Daily (every day)        |
    /// | W    | WS      | Weekly Monday            |
    /// | M    | MS      | Month Start (1st)        |
    /// | Q    | QS      | Quarter Start (1st)      |
    /// | Y    | A, AS, YS | Year Start (Jan 1st)   |
    /// | ME   |         | Month End (Last day)     |
    /// | QE   |         | Quarter End (Last day)   |
    /// | WF   |         | Weekly Friday            |
    /// | YE   | AE      | Year End (Dec 31st)      |
    ///
    /// # Arguments
    ///
    /// * `freq` - The frequency string slice (e.g., "D", "W", "ME").
    ///
    /// # Errors
    ///
    /// Returns an error if the string does not match any known frequency.
    fn from_str(freq: &str) -> Result<Self, Self::Err> {
        let r = match freq {
            "D" => DateFreq::Daily,
            "W" | "WS" => DateFreq::WeeklyMonday,
            "M" | "MS" => DateFreq::MonthStart,
            "Q" | "QS" => DateFreq::QuarterStart,
            "Y" | "A" | "AS" | "YS" => DateFreq::YearStart,
            "ME" => DateFreq::MonthEnd,
            "QE" => DateFreq::QuarterEnd,
            "WF" => DateFreq::WeeklyFriday,
            "YE" | "AE" => DateFreq::YearEnd,
            _ => return Err(format!("Invalid frequency specified: {}", freq).into()),
        };
        Ok(r)
    }
}

// --- DatesList Struct ---

/// Represents a list of calendar dates generated between a start and end date
/// at a specified frequency. Provides methods to retrieve the full list,
/// count, or dates grouped by period.
#[derive(Debug, Clone)]
pub struct DatesList {
    start_date_str: String,
    end_date_str: String,
    freq: DateFreq,
}

// Helper enum to represent the key for grouping dates into periods.
// Deriving traits for comparison and hashing allows using it as a HashMap key
// and for sorting groups chronologically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum GroupKey {
    Daily(NaiveDate),    // Group by the specific date (for Daily frequency)
    Weekly(i32, u32),    // Group by year and ISO week number
    Monthly(i32, u32),   // Group by year and month (1-12)
    Quarterly(i32, u32), // Group by year and quarter (1-4)
    Yearly(i32),         // Group by year
}

/// Represents a collection of calendar dates generated according to specific rules.
///
/// It can be defined either by a start and end date range or by a start date
/// and a fixed number of periods. It provides methods to retrieve the dates
/// as a flat list, count them, or group them by their natural period
/// (e.g., month, quarter).
///
/// This struct handles all calendar dates, including weekends.
///
/// ## Examples
///
/// **1. Using `DatesList::new` (Start and End Date):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use rustframe::utils::{DatesList, DateFreq}; // Assuming the crate/module is named 'dates'
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2023-11-01".to_string(); // Wednesday
/// let end_date = "2023-11-07".to_string();   // Tuesday
/// let freq = DateFreq::Daily;
///
/// let dates_list = DatesList::new(start_date, end_date, freq);
///
/// let expected_dates = vec![
///     NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(), // Wed
///     NaiveDate::from_ymd_opt(2023, 11, 2).unwrap(), // Thu
///     NaiveDate::from_ymd_opt(2023, 11, 3).unwrap(), // Fri
///     NaiveDate::from_ymd_opt(2023, 11, 4).unwrap(), // Sat
///     NaiveDate::from_ymd_opt(2023, 11, 5).unwrap(), // Sun
///     NaiveDate::from_ymd_opt(2023, 11, 6).unwrap(), // Mon
///     NaiveDate::from_ymd_opt(2023, 11, 7).unwrap(), // Tue
/// ];
///
/// assert_eq!(dates_list.list()?, expected_dates);
/// assert_eq!(dates_list.count()?, 7);
/// # Ok(())
/// # }
/// ```
///
/// **2. Using `DatesList::from_n_periods` (Start Date and Count):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{DatesList, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2024-02-28".to_string(); // Wednesday
/// let freq = DateFreq::WeeklyFriday;
/// let n_periods = 3;
///
/// let dates_list = DatesList::from_n_periods(start_date, freq, n_periods)?;
///
/// // The first Friday on or after 2024-02-28 is Mar 1.
/// // The next two Fridays are Mar 8 and Mar 15.
/// let expected_dates = vec![
///     NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
///     NaiveDate::from_ymd_opt(2024, 3, 8).unwrap(),
///     NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
/// ];
///
/// assert_eq!(dates_list.list()?, expected_dates);
/// assert_eq!(dates_list.count()?, 3);
/// assert_eq!(dates_list.start_date_str(), "2024-02-28"); // Keeps original start string
/// assert_eq!(dates_list.end_date_str(), "2024-03-15");   // End date is the last generated date
/// # Ok(())
/// # }
/// ```
///
/// **3. Using `groups()`:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{DatesList, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2023-11-20".to_string(); // Mon, Week 47
/// let end_date = "2023-12-08".to_string();   // Fri, Week 49
/// let freq = DateFreq::MonthEnd; // Find month-ends
///
/// let dates_list = DatesList::new(start_date, end_date, freq);
///
/// // Month ends >= Nov 20 and <= Dec 08: Nov 30
/// let groups = dates_list.groups()?;
///
/// assert_eq!(groups.len(), 1); // Only November's end date falls in the range
/// assert_eq!(groups[0], vec![NaiveDate::from_ymd_opt(2023, 11, 30).unwrap()]); // Nov 2023 group
/// Ok(())
/// }
/// ```
impl DatesList {
    /// Creates a new `DatesList` instance defined by a start and end date.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The inclusive start date as a string (e.g., "YYYY-MM-DD").
    /// * `end_date_str` - The inclusive end date as a string (e.g., "YYYY-MM-DD").
    /// * `freq` - The frequency for generating dates.
    pub fn new(start_date_str: String, end_date_str: String, freq: DateFreq) -> Self {
        DatesList {
            start_date_str,
            end_date_str,
            freq,
        }
    }

    /// Creates a new `DatesList` instance defined by a start date, frequency,
    /// and the number of periods (dates) to generate.
    ///
    /// This calculates the required dates using a `DatesGenerator` and determines
    /// the effective end date based on the last generated date.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The start date as a string (e.g., "YYYY-MM-DD"). The first generated date will be on or after this date.
    /// * `freq` - The frequency for generating dates.
    /// * `n_periods` - The exact number of dates to generate according to the frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `start_date_str` cannot be parsed.
    /// * `n_periods` is 0 (as this would result in an empty list and no defined end date).
    pub fn from_n_periods(
        start_date_str: String,
        freq: DateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if n_periods == 0 {
            return Err("n_periods must be greater than 0".into());
        }

        let start_date = NaiveDate::parse_from_str(&start_date_str, "%Y-%m-%d")?;

        // Use the generator to find all the dates
        let generator = DatesGenerator::new(start_date, freq, n_periods)?;
        let dates: Vec<NaiveDate> = generator.collect();

        // Should always have at least one date if n_periods > 0 and generator construction succeeded
        let last_date = dates
            .last()
            .ok_or("Generator failed to produce dates even though n_periods > 0")?;

        let end_date_str = last_date.format("%Y-%m-%d").to_string();

        Ok(DatesList {
            start_date_str, // Keep the original start date string
            end_date_str,
            freq,
        })
    }

    /// Returns the flat list of dates within the specified range and frequency.
    ///
    /// The list is guaranteed to be sorted chronologically.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn list(&self) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
        // Delegate the core logic to the internal helper function
        get_dates_list_with_freq(&self.start_date_str, &self.end_date_str, self.freq)
    }

    /// Returns the count of dates within the specified range and frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed (as it
    /// calls `list` internally).
    pub fn count(&self) -> Result<usize, Box<dyn Error>> {
        self.list().map(|list| list.len())
    }

    /// Returns a list of date lists, where each inner list contains dates
    /// belonging to the same period (determined by frequency).
    ///
    /// The outer list (groups) is sorted by period chronologically, and the
    /// inner lists (dates within groups) are also sorted chronologically.
    ///
    /// For `Daily` frequency, each date forms its own group. For `Weekly`
    /// frequencies, grouping is by ISO week number. For `Monthly`, `Quarterly`,
    /// and `Yearly` frequencies, grouping is by the respective period.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn groups(&self) -> Result<Vec<Vec<NaiveDate>>, Box<dyn Error>> {
        let dates = self.list()?;
        let mut groups: HashMap<GroupKey, Vec<NaiveDate>> = HashMap::new();

        for date in dates {
            let key = match self.freq {
                DateFreq::Daily => GroupKey::Daily(date),
                DateFreq::WeeklyMonday | DateFreq::WeeklyFriday => {
                    let iso_week = date.iso_week();
                    GroupKey::Weekly(iso_week.year(), iso_week.week())
                }
                DateFreq::MonthStart | DateFreq::MonthEnd => {
                    GroupKey::Monthly(date.year(), date.month())
                }
                DateFreq::QuarterStart | DateFreq::QuarterEnd => {
                    GroupKey::Quarterly(date.year(), month_to_quarter(date.month()))
                }
                DateFreq::YearStart | DateFreq::YearEnd => GroupKey::Yearly(date.year()),
            };
            groups.entry(key).or_insert_with(Vec::new).push(date);
        }

        let mut sorted_groups: Vec<(GroupKey, Vec<NaiveDate>)> = groups.into_iter().collect();
        sorted_groups.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        // Dates within groups are already sorted because they came from the sorted `self.list()`.
        let result_groups = sorted_groups.into_iter().map(|(_, dates)| dates).collect();
        Ok(result_groups)
    }

    /// Returns the start date parsed as a `NaiveDate`.
    ///
    /// # Errors
    ///
    /// Returns an error if the start date string is not in "YYYY-MM-DD" format.
    pub fn start_date(&self) -> Result<NaiveDate, Box<dyn Error>> {
        NaiveDate::parse_from_str(&self.start_date_str, "%Y-%m-%d").map_err(|e| e.into())
    }

    /// Returns the start date string.
    pub fn start_date_str(&self) -> &str {
        &self.start_date_str
    }

    /// Returns the end date parsed as a `NaiveDate`.
    ///
    /// # Errors
    ///
    /// Returns an error if the end date string is not in "YYYY-MM-DD" format.
    pub fn end_date(&self) -> Result<NaiveDate, Box<dyn Error>> {
        NaiveDate::parse_from_str(&self.end_date_str, "%Y-%m-%d").map_err(|e| e.into())
    }

    /// Returns the end date string.
    pub fn end_date_str(&self) -> &str {
        &self.end_date_str
    }

    /// Returns the frequency enum.
    pub fn freq(&self) -> DateFreq {
        self.freq
    }

    /// Returns the canonical string representation of the frequency.
    pub fn freq_str(&self) -> String {
        self.freq.to_string()
    }
}

// --- Dates Generator (Iterator) ---

/// An iterator that generates a sequence of calendar dates based on a start date,
/// frequency, and a specified number of periods.
///
/// This implements the `Iterator` trait, allowing generation of dates one by one.
/// It's useful when you need to process dates lazily or only need a fixed number
/// starting from a specific point, without necessarily defining an end date beforehand.
/// # Examples
///
/// **1. Basic Iteration (Month End):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{DatesGenerator, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
/// let start = NaiveDate::from_ymd_opt(2023, 12, 28).unwrap(); // Thursday
/// let freq = DateFreq::MonthEnd;
/// let n_periods = 4; // Dec '23, Jan '24, Feb '24, Mar '24
///
/// let mut generator = DatesGenerator::new(start, freq, n_periods)?;
///
/// // First month-end on or after 2023-12-28 is 2023-12-31
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2023, 12, 31).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 31).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 2, 29).unwrap())); // Leap year
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 3, 31).unwrap()));
/// assert_eq!(generator.next(), None); // Exhausted
/// Ok(())
/// }
/// ```
///
/// **2. Collecting into a Vec (Daily):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{DatesGenerator, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
/// let start = NaiveDate::from_ymd_opt(2024, 4, 29).unwrap(); // Monday
/// let freq = DateFreq::Daily;
/// let n_periods = 5;
///
/// let generator = DatesGenerator::new(start, freq, n_periods)?;
/// let dates: Vec<NaiveDate> = generator.collect();
///
/// let expected_dates = vec![
///     NaiveDate::from_ymd_opt(2024, 4, 29).unwrap(), // Mon
///     NaiveDate::from_ymd_opt(2024, 4, 30).unwrap(), // Tue
///     NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),  // Wed
///     NaiveDate::from_ymd_opt(2024, 5, 2).unwrap(),  // Thu
///     NaiveDate::from_ymd_opt(2024, 5, 3).unwrap(),  // Fri
/// ];
///
/// assert_eq!(dates, expected_dates);
/// Ok(())
/// }
/// ```
///
/// **3. Starting on the Exact Day (Weekly Monday):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{DatesGenerator, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
/// let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(); // Monday
/// let freq = DateFreq::WeeklyMonday;
/// let n_periods = 3;
///
/// let mut generator = DatesGenerator::new(start, freq, n_periods)?;
///
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 8).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()));
/// assert_eq!(generator.next(), None);
/// Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DatesGenerator {
    freq: DateFreq,
    periods_remaining: usize,
    // Stores the *next* date to be yielded by the iterator.
    next_date_candidate: Option<NaiveDate>,
}

impl DatesGenerator {
    /// Creates a new `DatesGenerator`.
    ///
    /// It calculates the first valid date based on the `start_date` and `freq`,
    /// which will be the first item yielded by the iterator.
    ///
    /// # Arguments
    ///
    /// * `start_date` - The date from which to start searching for the first valid date.
    /// * `freq` - The frequency for generating dates.
    /// * `n_periods` - The total number of dates to generate.
    ///
    /// # Errors
    ///
    /// Returns an error if initial date calculation fails (e.g., due to overflow, though unlikely).
    pub fn new(
        start_date: NaiveDate,
        freq: DateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let first_date = if n_periods > 0 {
            Some(find_first_date_on_or_after(start_date, freq)?)
        } else {
            None // No dates to generate if n_periods is 0
        };

        Ok(DatesGenerator {
            freq,
            periods_remaining: n_periods,
            next_date_candidate: first_date,
        })
    }
}

impl Iterator for DatesGenerator {
    type Item = NaiveDate;

    /// Returns the next date in the sequence, or `None` if `n_periods`
    /// dates have already been generated.
    fn next(&mut self) -> Option<Self::Item> {
        match self.next_date_candidate {
            Some(current_date) if self.periods_remaining > 0 => {
                // Prepare the *next* candidate for the subsequent call
                // We calculate the next date *before* decrementing periods_remaining
                // If find_next_date fails, we treat it as the end of the sequence.
                self.next_date_candidate = find_next_date(current_date, self.freq).ok();

                // Decrement the count *after* potentially getting the next date
                self.periods_remaining -= 1;

                // Return the stored current date
                Some(current_date)
            }
            _ => {
                // Exhausted or no initial date
                self.periods_remaining = 0; // Ensure it's 0
                self.next_date_candidate = None;
                None
            }
        }
    }
}

// --- Internal helper functions ---

/// Generates the flat list of dates for the given range and frequency.
/// Assumes the `collect_*` functions return sorted dates.
fn get_dates_list_with_freq(
    start_date_str: &str,
    end_date_str: &str,
    freq: DateFreq,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let start_date = NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d")?;
    let end_date = NaiveDate::parse_from_str(end_date_str, "%Y-%m-%d")?;

    if start_date > end_date {
        return Ok(Vec::new());
    }

    let dates = match freq {
        DateFreq::Daily => collect_daily(start_date, end_date)?,
        DateFreq::WeeklyMonday => collect_weekly(start_date, end_date, Weekday::Mon)?,
        DateFreq::WeeklyFriday => collect_weekly(start_date, end_date, Weekday::Fri)?,
        DateFreq::MonthStart => {
            collect_monthly(start_date, end_date, /*want_first_day=*/ true)?
        }
        DateFreq::MonthEnd => {
            collect_monthly(start_date, end_date, /*want_first_day=*/ false)?
        }
        DateFreq::QuarterStart => {
            collect_quarterly(start_date, end_date, /*want_first_day=*/ true)?
        }
        DateFreq::QuarterEnd => {
            collect_quarterly(start_date, end_date, /*want_first_day=*/ false)?
        }
        DateFreq::YearStart => collect_yearly(start_date, end_date, /*want_first_day=*/ true)?,
        DateFreq::YearEnd => collect_yearly(start_date, end_date, /*want_first_day=*/ false)?,
    };

    // The collect_* functions should now generate sorted dates directly.
    Ok(dates)
}

/* ---------------------- Low-Level Date Collection Functions (Internal) ---------------------- */
// These functions generate dates within a *range* [start_date, end_date]

/// Returns all calendar days day-by-day within the range.
fn collect_daily(
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let mut result = Vec::new();
    let mut current = start_date;
    while current <= end_date {
        result.push(current);
        current = current
            .succ_opt()
            .ok_or("Date overflow near end of supported range")?;
    }
    Ok(result)
}

/// Returns the specified `target_weekday` in each week within the range.
fn collect_weekly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    target_weekday: Weekday,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let mut result = Vec::new();
    let mut current = move_to_day_of_week_on_or_after(start_date, target_weekday)?;

    while current <= end_date {
        result.push(current);
        current = current
            .checked_add_signed(Duration::days(7))
            .ok_or("Date overflow adding 7 days")?;
    }
    Ok(result)
}

/// Returns either the first or last calendar day in each month of the range.
fn collect_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let mut result = Vec::new();
    let mut year = start_date.year();
    let mut month = start_date.month();

    let next_month =
        |(yr, mo): (i32, u32)| -> (i32, u32) { if mo == 12 { (yr + 1, 1) } else { (yr, mo + 1) } };

    loop {
        let candidate = if want_first_day {
            first_day_of_month(year, month)?
        } else {
            last_day_of_month(year, month)?
        };

        if candidate > end_date {
            break;
        }

        if candidate >= start_date {
            result.push(candidate);
        }

        if year > end_date.year() || (year == end_date.year() && month >= end_date.month()) {
            break;
        }

        let (ny, nm) = next_month((year, month));
        year = ny;
        month = nm;

        // Safety check for potential infinite loop, though unlikely with valid date logic
        if year > end_date.year() + 2 {
            return Err("Loop seems to exceed reasonable year range in collect_monthly".into());
        }
    }

    Ok(result)
}

/// Return either the first or last calendar day in each quarter of the range.
fn collect_quarterly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let mut result = Vec::new();
    let mut year = start_date.year();
    let mut q = month_to_quarter(start_date.month());

    loop {
        let candidate = if want_first_day {
            first_day_of_quarter(year, q)?
        } else {
            last_day_of_quarter(year, q)?
        };

        if candidate > end_date {
            break;
        }

        if candidate >= start_date {
            result.push(candidate);
        }

        let end_q = month_to_quarter(end_date.month());
        if year > end_date.year() || (year == end_date.year() && q >= end_q) {
            break;
        }

        if q == 4 {
            year += 1;
            q = 1;
        } else {
            q += 1;
        }
        // Safety check
        if year > end_date.year() + 2 {
            return Err("Loop seems to exceed reasonable year range in collect_quarterly".into());
        }
    }

    Ok(result)
}

/// Return either the first or last calendar day in each year of the range.
fn collect_yearly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    let mut result = Vec::new();
    let mut year = start_date.year();

    while year <= end_date.year() {
        let candidate = if want_first_day {
            first_day_of_year(year)?
        } else {
            last_day_of_year(year)?
        };

        if candidate >= start_date && candidate <= end_date {
            result.push(candidate);
        } else if candidate > end_date {
            // Optimization: If the candidate date is already past the end_date,
            // no subsequent year's candidate will be in range.
            break;
        }

        year = year.checked_add(1).ok_or("Year overflow")?;
    }
    Ok(result)
}

/* ---------------------- Core Date Utility Functions (Internal) ---------------------- */

/// Given a date and a `target_weekday`, returns the date that is the first
/// `target_weekday` on or after the given date.
fn move_to_day_of_week_on_or_after(
    date: NaiveDate,
    target: Weekday,
) -> Result<NaiveDate, Box<dyn Error>> {
    let mut current = date;
    while current.weekday() != target {
        current = current
            .succ_opt()
            .ok_or("Date overflow moving to next weekday")?;
    }
    Ok(current)
}

/// Return the first calendar day of the given (year, month).
fn first_day_of_month(year: i32, month: u32) -> Result<NaiveDate, Box<dyn Error>> {
    if !(1..=12).contains(&month) {
        return Err(format!("Invalid month: {}", month).into());
    }
    NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| format!("Invalid year-month combination: {}-{}", year, month).into())
}
/// Returns the number of days in a given month and year.
fn days_in_month(year: i32, month: u32) -> Result<u32, Box<dyn Error>> {
    if !(1..=12).contains(&month) {
        return Err(format!("Invalid month: {}", month).into());
    }
    let (ny, nm) = if month == 12 {
        (
            year.checked_add(1)
                .ok_or("Year overflow calculating next month")?,
            1,
        )
    } else {
        (year, month + 1)
    };
    // Use first_day_of_month which handles ymd creation errors
    let first_of_next = first_day_of_month(ny, nm)?;
    let last_of_this = first_of_next
        .pred_opt()
        .ok_or("Date underflow calculating last day of month")?;
    Ok(last_of_this.day())
}

/// Return the last calendar day of the given (year, month).
fn last_day_of_month(year: i32, month: u32) -> Result<NaiveDate, Box<dyn Error>> {
    // days_in_month now validates month and handles overflow
    let last_dom = days_in_month(year, month)?;
    NaiveDate::from_ymd_opt(year, month, last_dom)
        .ok_or_else(|| format!("Invalid year-month-day: {}-{}-{}", year, month, last_dom).into())
}

/// Converts a month number (1-12) to a quarter number (1-4).
/// Panics if month is invalid (should not happen with valid NaiveDate).
fn month_to_quarter(m: u32) -> u32 {
    match m {
        1..=3 => 1,
        4..=6 => 2,
        7..=9 => 3,
        10..=12 => 4,
        _ => panic!("Invalid month: {}", m), // Should only happen with programmer error
    }
}

/// Returns the 1st day of the month that starts a given quarter.
fn quarter_start_month(quarter: u32) -> Result<u32, Box<dyn Error>> {
    match quarter {
        1 => Ok(1),                                               // Jan
        2 => Ok(4),                                               // Apr
        3 => Ok(7),                                               // Jul
        4 => Ok(10),                                              // Oct
        _ => Err(format!("invalid quarter: {}", quarter).into()), // Return Err instead of panic
    }
}

/// Return the first calendar day in the given (year, quarter).
fn first_day_of_quarter(year: i32, quarter: u32) -> Result<NaiveDate, Box<dyn Error>> {
    // Propagate error from quarter_start_month
    let month = quarter_start_month(quarter)?;
    first_day_of_month(year, month)
}

/// Returns the last day of the month that ends a given quarter.
fn quarter_end_month(quarter: u32) -> Result<u32, Box<dyn Error>> {
    match quarter {
        1 => Ok(3),                                               // Mar
        2 => Ok(6),                                               // Jun
        3 => Ok(9),                                               // Sep
        4 => Ok(12),                                              // Dec
        _ => Err(format!("invalid quarter: {}", quarter).into()), // Return Err instead of panic
    }
}

/// Return the last calendar day in the given (year, quarter).
fn last_day_of_quarter(year: i32, quarter: u32) -> Result<NaiveDate, Box<dyn Error>> {
    // Propagate error from quarter_end_month
    let month = quarter_end_month(quarter)?;
    last_day_of_month(year, month)
}
/// Returns the first calendar day (Jan 1st) of the given year.
fn first_day_of_year(year: i32) -> Result<NaiveDate, Box<dyn Error>> {
    NaiveDate::from_ymd_opt(year, 1, 1)
        .ok_or_else(|| format!("Invalid year for Jan 1st: {}", year).into())
}

/// Returns the last calendar day (Dec 31st) of the given year.
fn last_day_of_year(year: i32) -> Result<NaiveDate, Box<dyn Error>> {
    NaiveDate::from_ymd_opt(year, 12, 31)
        .ok_or_else(|| format!("Invalid year for Dec 31st: {}", year).into())
}

// --- Generator Helper Functions ---

/// Finds the *first* valid date according to the frequency,
/// starting the search *on or after* the given `start_date`.
fn find_first_date_on_or_after(
    start_date: NaiveDate,
    freq: DateFreq,
) -> Result<NaiveDate, Box<dyn Error>> {
    match freq {
        DateFreq::Daily => Ok(start_date), // The first daily date is the start date itself
        DateFreq::WeeklyMonday => move_to_day_of_week_on_or_after(start_date, Weekday::Mon),
        DateFreq::WeeklyFriday => move_to_day_of_week_on_or_after(start_date, Weekday::Fri),
        DateFreq::MonthStart => {
            let mut candidate = first_day_of_month(start_date.year(), start_date.month())?;
            if candidate < start_date {
                let (next_y, next_m) = if start_date.month() == 12 {
                    (start_date.year().checked_add(1).ok_or("Year overflow")?, 1)
                } else {
                    (start_date.year(), start_date.month() + 1)
                };
                candidate = first_day_of_month(next_y, next_m)?;
            }
            Ok(candidate)
        }
        DateFreq::MonthEnd => {
            let mut candidate = last_day_of_month(start_date.year(), start_date.month())?;
            if candidate < start_date {
                let (next_y, next_m) = if start_date.month() == 12 {
                    (start_date.year().checked_add(1).ok_or("Year overflow")?, 1)
                } else {
                    (start_date.year(), start_date.month() + 1)
                };
                candidate = last_day_of_month(next_y, next_m)?;
            }
            Ok(candidate)
        }
        DateFreq::QuarterStart => {
            let current_q = month_to_quarter(start_date.month());
            let mut candidate = first_day_of_quarter(start_date.year(), current_q)?;
            if candidate < start_date {
                let (next_y, next_q) = if current_q == 4 {
                    (start_date.year().checked_add(1).ok_or("Year overflow")?, 1)
                } else {
                    (start_date.year(), current_q + 1)
                };
                candidate = first_day_of_quarter(next_y, next_q)?;
            }
            Ok(candidate)
        }
        DateFreq::QuarterEnd => {
            let current_q = month_to_quarter(start_date.month());
            let mut candidate = last_day_of_quarter(start_date.year(), current_q)?;
            if candidate < start_date {
                let (next_y, next_q) = if current_q == 4 {
                    (start_date.year().checked_add(1).ok_or("Year overflow")?, 1)
                } else {
                    (start_date.year(), current_q + 1)
                };
                candidate = last_day_of_quarter(next_y, next_q)?;
            }
            Ok(candidate)
        }
        DateFreq::YearStart => {
            let mut candidate = first_day_of_year(start_date.year())?;
            if candidate < start_date {
                candidate =
                    first_day_of_year(start_date.year().checked_add(1).ok_or("Year overflow")?)?;
            }
            Ok(candidate)
        }
        DateFreq::YearEnd => {
            let mut candidate = last_day_of_year(start_date.year())?;
            if candidate < start_date {
                candidate =
                    last_day_of_year(start_date.year().checked_add(1).ok_or("Year overflow")?)?;
            }
            Ok(candidate)
        }
    }
}

/// Finds the *next* valid date according to the frequency,
/// given the `current_date` (which is assumed to be a valid date previously generated).
fn find_next_date(current_date: NaiveDate, freq: DateFreq) -> Result<NaiveDate, Box<dyn Error>> {
    match freq {
        DateFreq::Daily => current_date
            .succ_opt()
            .ok_or_else(|| "Date overflow finding next daily".into()),
        DateFreq::WeeklyMonday | DateFreq::WeeklyFriday => current_date
            .checked_add_signed(Duration::days(7))
            .ok_or_else(|| "Date overflow adding 7 days".into()),
        DateFreq::MonthStart => {
            let (next_y, next_m) = if current_date.month() == 12 {
                (
                    current_date.year().checked_add(1).ok_or("Year overflow")?,
                    1,
                )
            } else {
                (current_date.year(), current_date.month() + 1)
            };
            first_day_of_month(next_y, next_m)
        }
        DateFreq::MonthEnd => {
            let (next_y, next_m) = if current_date.month() == 12 {
                (
                    current_date.year().checked_add(1).ok_or("Year overflow")?,
                    1,
                )
            } else {
                (current_date.year(), current_date.month() + 1)
            };
            last_day_of_month(next_y, next_m)
        }
        DateFreq::QuarterStart => {
            let current_q = month_to_quarter(current_date.month());
            let (next_y, next_q) = if current_q == 4 {
                (
                    current_date.year().checked_add(1).ok_or("Year overflow")?,
                    1,
                )
            } else {
                (current_date.year(), current_q + 1)
            };
            first_day_of_quarter(next_y, next_q)
        }
        DateFreq::QuarterEnd => {
            let current_q = month_to_quarter(current_date.month());
            let (next_y, next_q) = if current_q == 4 {
                (
                    current_date.year().checked_add(1).ok_or("Year overflow")?,
                    1,
                )
            } else {
                (current_date.year(), current_q + 1)
            };
            last_day_of_quarter(next_y, next_q)
        }
        DateFreq::YearStart => {
            first_day_of_year(current_date.year().checked_add(1).ok_or("Year overflow")?)
        }
        DateFreq::YearEnd => {
            last_day_of_year(current_date.year().checked_add(1).ok_or("Year overflow")?)
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, NaiveDate, Weekday}; // Make sure Duration is imported

    // Helper to create a NaiveDate for tests, expecting valid dates.
    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("Invalid date in test setup")
    }

    // --- DateFreq Tests ---

    #[test]
    fn test_datefreq_from_str() -> Result<(), Box<dyn Error>> {
        assert_eq!(DateFreq::from_str("D")?, DateFreq::Daily);
        assert_eq!("D".parse::<DateFreq>()?, DateFreq::Daily);
        assert_eq!(DateFreq::from_str("W")?, DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("WS")?, DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("M")?, DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("MS")?, DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("Q")?, DateFreq::QuarterStart);
        assert_eq!(DateFreq::from_str("QS")?, DateFreq::QuarterStart);
        assert_eq!(DateFreq::from_str("Y")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("A")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("AS")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("YS")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("ME")?, DateFreq::MonthEnd);
        assert_eq!(DateFreq::from_str("QE")?, DateFreq::QuarterEnd);
        assert_eq!(DateFreq::from_str("WF")?, DateFreq::WeeklyFriday);
        assert_eq!("WF".parse::<DateFreq>()?, DateFreq::WeeklyFriday);
        assert_eq!(DateFreq::from_str("YE")?, DateFreq::YearEnd);
        assert_eq!(DateFreq::from_str("AE")?, DateFreq::YearEnd);

        assert!(DateFreq::from_str("INVALID").is_err());
        assert!("INVALID".parse::<DateFreq>().is_err());
        let err = DateFreq::from_str("INVALID").unwrap_err();
        assert_eq!(err.to_string(), "Invalid frequency specified: INVALID");

        Ok(())
    }

    #[test]
    fn test_datefreq_to_string() {
        assert_eq!(DateFreq::Daily.to_string(), "D");
        assert_eq!(DateFreq::WeeklyMonday.to_string(), "W");
        assert_eq!(DateFreq::MonthStart.to_string(), "M");
        assert_eq!(DateFreq::QuarterStart.to_string(), "Q");
        assert_eq!(DateFreq::YearStart.to_string(), "Y");
        assert_eq!(DateFreq::MonthEnd.to_string(), "ME");
        assert_eq!(DateFreq::QuarterEnd.to_string(), "QE");
        assert_eq!(DateFreq::WeeklyFriday.to_string(), "WF");
        assert_eq!(DateFreq::YearEnd.to_string(), "YE");
    }

    #[test]
    fn test_datefreq_from_string() -> Result<(), Box<dyn Error>> {
        assert_eq!(DateFreq::from_string("D".to_string())?, DateFreq::Daily);
        assert!(DateFreq::from_string("INVALID".to_string()).is_err());
        Ok(())
    }

    #[test]
    fn test_datefreq_agg_type() {
        assert_eq!(DateFreq::Daily.agg_type(), AggregationType::Start);
        assert_eq!(DateFreq::WeeklyMonday.agg_type(), AggregationType::Start);
        assert_eq!(DateFreq::MonthStart.agg_type(), AggregationType::Start);
        assert_eq!(DateFreq::QuarterStart.agg_type(), AggregationType::Start);
        assert_eq!(DateFreq::YearStart.agg_type(), AggregationType::Start);

        assert_eq!(DateFreq::WeeklyFriday.agg_type(), AggregationType::End);
        assert_eq!(DateFreq::MonthEnd.agg_type(), AggregationType::End);
        assert_eq!(DateFreq::QuarterEnd.agg_type(), AggregationType::End);
        assert_eq!(DateFreq::YearEnd.agg_type(), AggregationType::End);
    }

    // --- DatesList Property Tests ---

    #[test]
    fn test_dates_list_properties_new() -> Result<(), Box<dyn Error>> {
        let start_str = "2023-01-01".to_string();
        let end_str = "2023-12-31".to_string();
        let freq = DateFreq::QuarterEnd;
        let dates_list = DatesList::new(start_str.clone(), end_str.clone(), freq);

        assert_eq!(dates_list.start_date_str(), start_str);
        assert_eq!(dates_list.end_date_str(), end_str);
        assert_eq!(dates_list.freq(), freq);
        assert_eq!(dates_list.freq_str(), "QE");
        assert_eq!(dates_list.start_date()?, date(2023, 1, 1));
        assert_eq!(dates_list.end_date()?, date(2023, 12, 31));

        Ok(())
    }

    #[test]
    fn test_dates_list_properties_from_n_periods() -> Result<(), Box<dyn Error>> {
        let start_str = "2023-01-01".to_string(); // Sunday
        let freq = DateFreq::Daily;
        let n_periods = 5; // Expect: Jan 1, 2, 3, 4, 5
        let dates_list = DatesList::from_n_periods(start_str.clone(), freq, n_periods)?;

        assert_eq!(dates_list.start_date_str(), start_str);
        assert_eq!(dates_list.end_date_str(), "2023-01-05");
        assert_eq!(dates_list.freq(), freq);
        assert_eq!(dates_list.freq_str(), "D");
        assert_eq!(dates_list.start_date()?, date(2023, 1, 1));
        assert_eq!(dates_list.end_date()?, date(2023, 1, 5));

        assert_eq!(
            dates_list.list()?,
            vec![
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 4),
                date(2023, 1, 5)
            ]
        );
        assert_eq!(dates_list.count()?, 5);

        Ok(())
    }

    #[test]
    fn test_dates_list_from_n_periods_zero_periods() {
        let start_str = "2023-01-01".to_string();
        let freq = DateFreq::Daily;
        let n_periods = 0;
        let result = DatesList::from_n_periods(start_str.clone(), freq, n_periods);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "n_periods must be greater than 0"
        );
    }

    // test_dates_list_from_n_periods_fail_get_last removed as it was flawed

    #[test]
    fn test_dates_list_from_n_periods_invalid_start_date() {
        let start_str = "invalid-date".to_string();
        let freq = DateFreq::Daily;
        let n_periods = 5;
        let result = DatesList::from_n_periods(start_str.clone(), freq, n_periods);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("input contains invalid characters") // Error from NaiveDate::parse_from_str
        );
    }

    #[test]
    fn test_dates_list_invalid_date_string_new() {
        let list_start_invalid = DatesList::new(
            "invalid-date".to_string(),
            "2023-12-31".to_string(),
            DateFreq::Daily,
        );
        assert!(list_start_invalid.list().is_err());
        assert!(list_start_invalid.count().is_err());
        assert!(list_start_invalid.groups().is_err());
        assert!(list_start_invalid.start_date().is_err());
        assert!(list_start_invalid.end_date().is_ok()); // End date is valid

        let list_end_invalid = DatesList::new(
            "2023-01-01".to_string(),
            "invalid-date".to_string(),
            DateFreq::Daily,
        );
        assert!(list_end_invalid.list().is_err());
        assert!(list_end_invalid.count().is_err());
        assert!(list_end_invalid.groups().is_err());
        assert!(list_end_invalid.start_date().is_ok()); // Start date is valid
        assert!(list_end_invalid.end_date().is_err());
    }

    // --- DatesList Core Logic Tests (list, count) ---

    #[test]
    fn test_dates_list_daily_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-05".to_string(), // Sun
            DateFreq::Daily,
        );
        let list = dates_list.list()?;
        assert_eq!(list.len(), 5);
        assert_eq!(
            list,
            vec![
                date(2023, 11, 1),
                date(2023, 11, 2),
                date(2023, 11, 3),
                date(2023, 11, 4),
                date(2023, 11, 5)
            ]
        );
        assert_eq!(dates_list.count()?, 5);
        Ok(())
    }

    #[test]
    fn test_dates_list_weekly_monday_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-10-30".to_string(), // Mon
            "2023-11-13".to_string(), // Mon
            DateFreq::WeeklyMonday,
        );
        let list = dates_list.list()?;
        // Mondays in range: Oct 30, Nov 6, Nov 13
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2023, 10, 30), date(2023, 11, 6), date(2023, 11, 13)]
        );
        assert_eq!(dates_list.count()?, 3);
        Ok(())
    }

    #[test]
    fn test_dates_list_weekly_friday_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-17".to_string(), // Fri
            DateFreq::WeeklyFriday,
        );
        let list = dates_list.list()?;
        // Fridays in range: Nov 3, Nov 10, Nov 17
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2023, 11, 3), date(2023, 11, 10), date(2023, 11, 17)]
        );
        assert_eq!(dates_list.count()?, 3);
        Ok(())
    }

    #[test]
    fn test_dates_list_month_start_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-11-15".to_string(), // Mid-Nov
            "2024-02-01".to_string(), // Feb 1st
            DateFreq::MonthStart,
        );
        let list = dates_list.list()?;
        // Month starts >= Nov 15 and <= Feb 1: Dec 1, Jan 1, Feb 1
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2023, 12, 1), date(2024, 1, 1), date(2024, 2, 1)]
        );
        assert_eq!(dates_list.count()?, 3);
        Ok(())
    }

    #[test]
    fn test_dates_list_month_end_list_leap() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2024-01-15".to_string(), // Mid-Jan
            "2024-03-31".to_string(), // Mar 31st
            DateFreq::MonthEnd,
        );
        let list = dates_list.list()?;
        // Month ends >= Jan 15 and <= Mar 31: Jan 31, Feb 29 (leap), Mar 31
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]
        );
        assert_eq!(dates_list.count()?, 3);
        Ok(())
    }

    #[test]
    fn test_dates_list_quarter_start_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-08-01".to_string(), // Mid Q3
            "2024-04-01".to_string(), // Start Q2
            DateFreq::QuarterStart,
        );
        let list = dates_list.list()?;
        // Quarter starts >= Aug 1 '23 and <= Apr 1 '24: Oct 1 '23, Jan 1 '24, Apr 1 '24
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2023, 10, 1), date(2024, 1, 1), date(2024, 4, 1)]
        );
        assert_eq!(dates_list.count()?, 3);
        Ok(())
    }

    #[test]
    fn test_dates_list_quarter_end_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-03-31".to_string(), // End Q1
            "2023-12-31".to_string(), // End Q4
            DateFreq::QuarterEnd,
        );
        let list = dates_list.list()?;
        // Quarter ends >= Mar 31 and <= Dec 31: Mar 31, Jun 30, Sep 30, Dec 31
        assert_eq!(list.len(), 4);
        assert_eq!(
            list,
            vec![
                date(2023, 3, 31),
                date(2023, 6, 30),
                date(2023, 9, 30),
                date(2023, 12, 31)
            ]
        );
        assert_eq!(dates_list.count()?, 4);
        Ok(())
    }

    #[test]
    fn test_dates_list_year_start_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-06-01".to_string(), // Mid 2023
            "2025-01-01".to_string(), // Start 2025
            DateFreq::YearStart,
        );
        let list = dates_list.list()?;
        // Year starts >= Jun 1 '23 and <= Jan 1 '25: Jan 1 '24, Jan 1 '25
        assert_eq!(list.len(), 2);
        assert_eq!(list, vec![date(2024, 1, 1), date(2025, 1, 1)]);
        assert_eq!(dates_list.count()?, 2);
        Ok(())
    }

    #[test]
    fn test_dates_list_year_end_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2022-01-01".to_string(), // Start 2022
            "2024-03-31".to_string(), // Q1 2024
            DateFreq::YearEnd,
        );
        let list = dates_list.list()?;
        // Year ends >= Jan 1 '22 and <= Mar 31 '24: Dec 31 '22, Dec 31 '23
        assert_eq!(list.len(), 2);
        assert_eq!(list, vec![date(2022, 12, 31), date(2023, 12, 31)]);
        assert_eq!(dates_list.count()?, 2);
        Ok(())
    }

    #[test]
    fn test_dates_list_empty_range_list() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-12-31".to_string(),
            "2023-01-01".to_string(), // End date before start date
            DateFreq::Daily,
        );
        let list = dates_list.list()?;
        assert!(list.is_empty());
        assert_eq!(dates_list.count()?, 0);
        Ok(())
    }

    // --- Tests for groups() method ---

    #[test]
    fn test_dates_list_groups_monthly_end() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-10-15".to_string(), // Mid-Oct
            "2024-01-15".to_string(), // Mid-Jan
            DateFreq::MonthEnd,
        );
        let groups = dates_list.groups()?;
        // Month Ends >= Oct 15 '23 and <= Jan 15 '24: Oct 31, Nov 30, Dec 31
        assert_eq!(groups.len(), 3);
        // Key order: Monthly(2023, 10), Monthly(2023, 11), Monthly(2023, 12)
        assert_eq!(groups[0], vec![date(2023, 10, 31)]);
        assert_eq!(groups[1], vec![date(2023, 11, 30)]);
        assert_eq!(groups[2], vec![date(2023, 12, 31)]);
        Ok(())
    }

    #[test]
    fn test_dates_list_groups_daily() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-03".to_string(), // Fri
            DateFreq::Daily,
        );
        let groups = dates_list.groups()?;
        // Dates: Nov 1, Nov 2, Nov 3. Each gets own group.
        assert_eq!(groups.len(), 3);
        // Key order: Daily(2023-11-01), Daily(2023-11-02), Daily(2023-11-03)
        assert_eq!(groups[0], vec![date(2023, 11, 1)]);
        assert_eq!(groups[1], vec![date(2023, 11, 2)]);
        assert_eq!(groups[2], vec![date(2023, 11, 3)]);
        Ok(())
    }

    #[test]
    fn test_dates_list_groups_weekly_friday() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-11-01".to_string(), // Wed (ISO Week 44)
            "2023-11-15".to_string(), // Wed (ISO Week 46)
            DateFreq::WeeklyFriday,
        );
        let groups = dates_list.groups()?;
        // Fridays in range: Nov 3 (W44), Nov 10 (W45)
        assert_eq!(groups.len(), 2);
        // Key order: Weekly(2023, 44), Weekly(2023, 45)
        assert_eq!(groups[0], vec![date(2023, 11, 3)]);
        assert_eq!(groups[1], vec![date(2023, 11, 10)]);
        Ok(())
    }

    #[test]
    fn test_dates_list_groups_quarterly_start() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-08-01".to_string(), // Start Q3
            "2024-05-01".to_string(), // Start Q2
            DateFreq::QuarterStart,
        );
        let groups = dates_list.groups()?;
        // Quarter starts >= Aug 1 '23 and <= May 1 '24: Oct 1 '23, Jan 1 '24, Apr 1 '24
        assert_eq!(groups.len(), 3);
        // Key order: Quarterly(2023, 4), Quarterly(2024, 1), Quarterly(2024, 2)
        assert_eq!(groups[0], vec![date(2023, 10, 1)]);
        assert_eq!(groups[1], vec![date(2024, 1, 1)]);
        assert_eq!(groups[2], vec![date(2024, 4, 1)]);
        Ok(())
    }

    #[test]
    fn test_dates_list_groups_yearly_end() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2022-01-01".to_string(), // Start 2022
            "2024-03-31".to_string(), // Q1 2024
            DateFreq::YearEnd,
        );
        let groups = dates_list.groups()?;
        // Year ends >= Jan 1 '22 and <= Mar 31 '24: Dec 31 '22, Dec 31 '23
        assert_eq!(groups.len(), 2);
        // Key order: Yearly(2022), Yearly(2023)
        assert_eq!(groups[0], vec![date(2022, 12, 31)]);
        assert_eq!(groups[1], vec![date(2023, 12, 31)]);
        Ok(())
    }

    #[test]
    fn test_dates_list_groups_empty_range() -> Result<(), Box<dyn Error>> {
        let dates_list = DatesList::new(
            "2023-12-31".to_string(),
            "2023-01-01".to_string(), // End < Start
            DateFreq::Daily,
        );
        let groups = dates_list.groups()?;
        assert!(groups.is_empty());
        Ok(())
    }

    // --- Tests for internal helper functions ---

    #[test]
    fn test_move_to_day_of_week_on_or_after() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            move_to_day_of_week_on_or_after(date(2023, 11, 6), Weekday::Mon)?,
            date(2023, 11, 6)
        );
        assert_eq!(
            move_to_day_of_week_on_or_after(date(2023, 11, 8), Weekday::Fri)?,
            date(2023, 11, 10)
        );
        assert_eq!(
            move_to_day_of_week_on_or_after(date(2023, 11, 11), Weekday::Mon)?, // Sat -> Mon
            date(2023, 11, 13)
        );
        assert_eq!(
            move_to_day_of_week_on_or_after(date(2023, 11, 10), Weekday::Mon)?, // Fri -> Mon
            date(2023, 11, 13)
        );
        // Test near max date (ensure it doesn't panic easily, though overflow is possible)
        // MAX - 7 days guarantees we have room to move forward
        let near_max = NaiveDate::MAX - Duration::days(7);
        assert!(move_to_day_of_week_on_or_after(near_max, Weekday::Sun).is_ok());
        // Test overflow case - starting at MAX, moving forward fails if MAX is not target
        if NaiveDate::MAX.weekday() != Weekday::Sun {
            assert!(move_to_day_of_week_on_or_after(NaiveDate::MAX, Weekday::Sun).is_err());
        } else {
            // If MAX is the target, it should succeed
            assert!(move_to_day_of_week_on_or_after(NaiveDate::MAX, Weekday::Sun).is_ok());
            // And trying to move *past* it should fail
            let day_before = NaiveDate::MAX - Duration::days(1);
            let target_day_after = NaiveDate::MAX.weekday().succ(); // Day after MAX's weekday
            assert!(move_to_day_of_week_on_or_after(day_before, target_day_after).is_err()); // Moving past MAX fails
        }

        Ok(())
    }

    #[test]
    fn test_first_day_of_month() -> Result<(), Box<dyn Error>> {
        assert_eq!(first_day_of_month(2023, 11)?, date(2023, 11, 1));
        assert_eq!(first_day_of_month(2024, 2)?, date(2024, 2, 1));
        assert!(first_day_of_month(2023, 0).is_err()); // Invalid month 0
        assert!(first_day_of_month(2023, 13).is_err()); // Invalid month 13
        Ok(())
    }

    #[test]
    fn test_days_in_month() -> Result<(), Box<dyn Error>> {
        assert_eq!(days_in_month(2023, 1)?, 31);
        assert_eq!(days_in_month(2023, 2)?, 28);
        assert_eq!(days_in_month(2024, 2)?, 29); // Leap
        assert_eq!(days_in_month(2023, 4)?, 30);
        assert_eq!(days_in_month(2023, 12)?, 31);
        assert!(days_in_month(2023, 0).is_err()); // Invalid month 0
        assert!(days_in_month(2023, 13).is_err()); // Invalid month 13
        // Test near max date year overflow - Use MAX.year()
        assert!(days_in_month(NaiveDate::MAX.year(), 12).is_err());
        Ok(())
    }

    #[test]
    fn test_last_day_of_month() -> Result<(), Box<dyn Error>> {
        assert_eq!(last_day_of_month(2023, 11)?, date(2023, 11, 30));
        assert_eq!(last_day_of_month(2024, 2)?, date(2024, 2, 29)); // Leap
        assert_eq!(last_day_of_month(2023, 12)?, date(2023, 12, 31));
        assert!(last_day_of_month(2023, 0).is_err()); // Invalid month 0
        assert!(last_day_of_month(2023, 13).is_err()); // Invalid month 13
        // Test near max date year overflow - use MAX.year()
        assert!(last_day_of_month(NaiveDate::MAX.year(), 12).is_err());
        Ok(())
    }

    #[test]
    fn test_month_to_quarter() {
        assert_eq!(month_to_quarter(1), 1);
        assert_eq!(month_to_quarter(3), 1);
        assert_eq!(month_to_quarter(4), 2);
        assert_eq!(month_to_quarter(6), 2);
        assert_eq!(month_to_quarter(7), 3);
        assert_eq!(month_to_quarter(9), 3);
        assert_eq!(month_to_quarter(10), 4);
        assert_eq!(month_to_quarter(12), 4);
    }

    #[test]
    #[should_panic(expected = "Invalid month: 0")]
    fn test_month_to_quarter_invalid_low() {
        month_to_quarter(0);
    }
    #[test]
    #[should_panic(expected = "Invalid month: 13")]
    fn test_month_to_quarter_invalid_high() {
        month_to_quarter(13);
    }

    #[test]
    fn test_quarter_start_month() {
        assert_eq!(quarter_start_month(1).unwrap(), 1);
        assert_eq!(quarter_start_month(2).unwrap(), 4);
        assert_eq!(quarter_start_month(3).unwrap(), 7);
        assert_eq!(quarter_start_month(4).unwrap(), 10);
        assert!(quarter_start_month(0).is_err());
        assert!(quarter_start_month(5).is_err());
    }

    #[test]
    fn test_first_day_of_quarter() -> Result<(), Box<dyn Error>> {
        assert_eq!(first_day_of_quarter(2023, 1)?, date(2023, 1, 1));
        assert_eq!(first_day_of_quarter(2023, 2)?, date(2023, 4, 1));
        assert_eq!(first_day_of_quarter(2023, 3)?, date(2023, 7, 1));
        assert_eq!(first_day_of_quarter(2023, 4)?, date(2023, 10, 1));
        assert!(first_day_of_quarter(2023, 5).is_err()); // Invalid quarter
        Ok(())
    }

    #[test]
    fn test_quarter_end_month() {
        assert_eq!(quarter_end_month(1).unwrap(), 3);
        assert_eq!(quarter_end_month(2).unwrap(), 6);
        assert_eq!(quarter_end_month(3).unwrap(), 9);
        assert_eq!(quarter_end_month(4).unwrap(), 12);
        assert!(quarter_end_month(0).is_err());
        assert!(quarter_end_month(5).is_err());
    }

    #[test]
    fn test_last_day_of_quarter() -> Result<(), Box<dyn Error>> {
        assert_eq!(last_day_of_quarter(2023, 1)?, date(2023, 3, 31));
        assert_eq!(last_day_of_quarter(2023, 2)?, date(2023, 6, 30));
        assert_eq!(last_day_of_quarter(2023, 3)?, date(2023, 9, 30));
        assert_eq!(last_day_of_quarter(2023, 4)?, date(2023, 12, 31));
        assert_eq!(last_day_of_quarter(2024, 1)?, date(2024, 3, 31)); // Leap year doesn't affect March end
        assert!(last_day_of_quarter(2023, 5).is_err()); // Invalid quarter
        // Test overflow propagation - use MAX.year()
        assert!(last_day_of_quarter(NaiveDate::MAX.year(), 4).is_err());
        Ok(())
    }

    #[test]
    fn test_first_day_of_year() -> Result<(), Box<dyn Error>> {
        assert_eq!(first_day_of_year(2023)?, date(2023, 1, 1));
        assert_eq!(first_day_of_year(2024)?, date(2024, 1, 1));
        // Test MAX year - should be ok for Jan 1
        assert!(first_day_of_year(NaiveDate::MAX.year()).is_ok());
        Ok(())
    }

    #[test]
    fn test_last_day_of_year() -> Result<(), Box<dyn Error>> {
        assert_eq!(last_day_of_year(2023)?, date(2023, 12, 31));
        assert_eq!(last_day_of_year(2024)?, date(2024, 12, 31)); // Leap year doesn't affect Dec 31st existence
        // Test MAX year - should be okay since MAX is Dec 31
        assert_eq!(last_day_of_year(NaiveDate::MAX.year())?, NaiveDate::MAX);
        Ok(())
    }

    // Overflow tests for collect_* removed as they were misleading

    // --- Tests for Generator Helper Functions ---

    #[test]
    fn test_find_first_date_on_or_after() -> Result<(), Box<dyn Error>> {
        // Daily
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 8), DateFreq::Daily)?,
            date(2023, 11, 8)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 11), DateFreq::Daily)?,
            date(2023, 11, 11)
        ); // Sat -> Sat

        // Weekly Mon
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 8), DateFreq::WeeklyMonday)?,
            date(2023, 11, 13)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 13), DateFreq::WeeklyMonday)?,
            date(2023, 11, 13)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 12), DateFreq::WeeklyMonday)?,
            date(2023, 11, 13)
        ); // Sun -> Mon

        // Weekly Fri
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 8), DateFreq::WeeklyFriday)?,
            date(2023, 11, 10)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 10), DateFreq::WeeklyFriday)?,
            date(2023, 11, 10)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 11), DateFreq::WeeklyFriday)?,
            date(2023, 11, 17)
        ); // Sat -> Next Fri

        // Month Start
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 1), DateFreq::MonthStart)?,
            date(2023, 11, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 10, 15), DateFreq::MonthStart)?,
            date(2023, 11, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 12, 15), DateFreq::MonthStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 10, 1), DateFreq::MonthStart)?,
            date(2023, 10, 1)
        ); // Oct 1 -> Oct 1

        // Month End
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 30), DateFreq::MonthEnd)?,
            date(2023, 11, 30)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 15), DateFreq::MonthEnd)?,
            date(2023, 11, 30)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 12, 31), DateFreq::MonthEnd)?,
            date(2023, 12, 31)
        ); // Dec 31 -> Dec 31
        assert_eq!(
            find_first_date_on_or_after(date(2024, 2, 15), DateFreq::MonthEnd)?,
            date(2024, 2, 29)
        ); // Mid Feb (Leap) -> Feb 29
        assert_eq!(
            find_first_date_on_or_after(date(2024, 2, 29), DateFreq::MonthEnd)?,
            date(2024, 2, 29)
        ); // Feb 29 -> Feb 29

        // Quarter Start
        assert_eq!(
            find_first_date_on_or_after(date(2023, 10, 1), DateFreq::QuarterStart)?,
            date(2023, 10, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 8, 15), DateFreq::QuarterStart)?,
            date(2023, 10, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 11, 15), DateFreq::QuarterStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 1, 1), DateFreq::QuarterStart)?,
            date(2023, 1, 1)
        );

        // Quarter End
        assert_eq!(
            find_first_date_on_or_after(date(2023, 9, 30), DateFreq::QuarterEnd)?,
            date(2023, 9, 30)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 8, 15), DateFreq::QuarterEnd)?,
            date(2023, 9, 30)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 10, 15), DateFreq::QuarterEnd)?,
            date(2023, 12, 31)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 12, 31), DateFreq::QuarterEnd)?,
            date(2023, 12, 31)
        );

        // Year Start
        assert_eq!(
            find_first_date_on_or_after(date(2024, 1, 1), DateFreq::YearStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 6, 15), DateFreq::YearStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 1, 1), DateFreq::YearStart)?,
            date(2023, 1, 1)
        );

        // Year End
        assert_eq!(
            find_first_date_on_or_after(date(2023, 12, 31), DateFreq::YearEnd)?,
            date(2023, 12, 31)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2023, 6, 15), DateFreq::YearEnd)?,
            date(2023, 12, 31)
        );
        assert_eq!(
            find_first_date_on_or_after(date(2022, 12, 31), DateFreq::YearEnd)?,
            date(2022, 12, 31)
        );

        // --- Test Overflow Cases near MAX ---
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::Daily).is_ok()); // Daily starting at MAX is MAX

        // Weekly: depends if MAX is the target day. If not, succ() fails.
        if NaiveDate::MAX.weekday() != Weekday::Mon {
            assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::WeeklyMonday).is_err());
        } else {
            assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::WeeklyMonday).is_ok());
        }
        // Month Start: MAX is Dec 31. find_first for MonthStart at MAX tries month=12, candidate=Dec 1. candidate < MAX is true.
        // Tries next month: Jan (MAX_YEAR+1), which fails in checked_add.
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::MonthStart).is_err());

        // Month End: MAX is Dec 31. find_first for MonthEnd at MAX tries month=12, calls last_day_of_month(MAX_YEAR, 12).
        // last_day_of_month -> days_in_month -> first_day_of_month(MAX_YEAR+1, 1) -> fails.
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::MonthEnd).is_err());

        // Quarter Start: MAX is Dec 31 (Q4). Tries Q4 start (Oct 1). candidate < MAX is true. Tries next Q (Q1 MAX+1), fails.
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::QuarterStart).is_err());

        // Quarter End: MAX is Dec 31 (Q4). Tries Q4 end (Dec 31). Calls last_day_of_quarter(MAX_YEAR, 4).
        // last_day_of_quarter -> last_day_of_month(MAX_YEAR, 12) -> fails.
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::QuarterEnd).is_err());

        // Year Start: MAX is Dec 31. Tries YearStart(MAX_YEAR) (Jan 1). candidate < MAX is true. Tries next year (MAX_YEAR+1), fails.
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::YearStart).is_err());

        // Year End: MAX is Dec 31. Tries YearEnd(MAX_YEAR). Calls last_day_of_year(MAX_YEAR). Returns Ok(MAX). candidate < MAX is false. Returns Ok(MAX).
        assert!(find_first_date_on_or_after(NaiveDate::MAX, DateFreq::YearEnd).is_ok());

        Ok(())
    }

    #[test]
    fn test_find_next_date() -> Result<(), Box<dyn Error>> {
        // Daily
        assert_eq!(
            find_next_date(date(2023, 11, 8), DateFreq::Daily)?,
            date(2023, 11, 9)
        );
        assert_eq!(
            find_next_date(date(2023, 11, 10), DateFreq::Daily)?,
            date(2023, 11, 11)
        ); // Fri -> Sat

        // Weekly Mon
        assert_eq!(
            find_next_date(date(2023, 11, 13), DateFreq::WeeklyMonday)?,
            date(2023, 11, 20)
        );

        // Weekly Fri
        assert_eq!(
            find_next_date(date(2023, 11, 10), DateFreq::WeeklyFriday)?,
            date(2023, 11, 17)
        );

        // Month Start
        assert_eq!(
            find_next_date(date(2023, 11, 1), DateFreq::MonthStart)?,
            date(2023, 12, 1)
        );
        assert_eq!(
            find_next_date(date(2023, 12, 1), DateFreq::MonthStart)?,
            date(2024, 1, 1)
        );

        // Month End
        assert_eq!(
            find_next_date(date(2023, 10, 31), DateFreq::MonthEnd)?,
            date(2023, 11, 30)
        );
        assert_eq!(
            find_next_date(date(2024, 1, 31), DateFreq::MonthEnd)?,
            date(2024, 2, 29)
        ); // Jan -> Feb (Leap)
        assert_eq!(
            find_next_date(date(2024, 2, 29), DateFreq::MonthEnd)?,
            date(2024, 3, 31)
        ); // Feb -> Mar

        // Quarter Start
        assert_eq!(
            find_next_date(date(2023, 10, 1), DateFreq::QuarterStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_next_date(date(2024, 1, 1), DateFreq::QuarterStart)?,
            date(2024, 4, 1)
        );

        // Quarter End
        assert_eq!(
            find_next_date(date(2023, 9, 30), DateFreq::QuarterEnd)?,
            date(2023, 12, 31)
        );
        assert_eq!(
            find_next_date(date(2023, 12, 31), DateFreq::QuarterEnd)?,
            date(2024, 3, 31)
        );

        // Year Start
        assert_eq!(
            find_next_date(date(2023, 1, 1), DateFreq::YearStart)?,
            date(2024, 1, 1)
        );
        assert_eq!(
            find_next_date(date(2024, 1, 1), DateFreq::YearStart)?,
            date(2025, 1, 1)
        );

        // Year End
        assert_eq!(
            find_next_date(date(2022, 12, 31), DateFreq::YearEnd)?,
            date(2023, 12, 31)
        );
        assert_eq!(
            find_next_date(date(2023, 12, 31), DateFreq::YearEnd)?,
            date(2024, 12, 31)
        );

        // --- Test Overflow Cases near MAX ---
        assert!(find_next_date(NaiveDate::MAX, DateFreq::Daily).is_err());
        assert!(
            find_next_date(NaiveDate::MAX - Duration::days(6), DateFreq::WeeklyMonday).is_err()
        );

        // Test finding next month start after Dec MAX_YEAR -> Jan (MAX_YEAR+1) (fail)
        assert!(find_next_date(date(NaiveDate::MAX.year(), 12, 1), DateFreq::MonthStart).is_err());

        // Test finding next month end after Nov MAX_YEAR -> Dec MAX_YEAR (fails because last_day_of_month(MAX, 12) fails)
        let nov_end_max_year = last_day_of_month(NaiveDate::MAX.year(), 11)?;
        assert!(find_next_date(nov_end_max_year, DateFreq::MonthEnd).is_err());

        // Test finding next month end after Dec MAX_YEAR -> Jan (MAX_YEAR+1) (fail)
        // The call last_day_of_month(MAX_YEAR + 1, 1) fails
        assert!(find_next_date(NaiveDate::MAX, DateFreq::MonthEnd).is_err());

        // Test finding next quarter start after Q4 MAX_YEAR -> Q1 (MAX_YEAR+1) (fail)
        assert!(
            find_next_date(
                first_day_of_quarter(NaiveDate::MAX.year(), 4)?,
                DateFreq::QuarterStart
            )
            .is_err()
        );

        // Test finding next quarter end after Q3 MAX_YEAR -> Q4 MAX_YEAR (fails because last_day_of_quarter(MAX, 4) fails)
        let q3_end_max_year = last_day_of_quarter(NaiveDate::MAX.year(), 3)?;
        assert!(find_next_date(q3_end_max_year, DateFreq::QuarterEnd).is_err());

        // Test finding next quarter end after Q4 MAX_YEAR -> Q1 (MAX_YEAR+1) (fail)
        // The call last_day_of_quarter(MAX_YEAR + 1, 1) fails
        assert!(find_next_date(NaiveDate::MAX, DateFreq::QuarterEnd).is_err());

        // Test finding next year start after Jan 1 MAX_YEAR -> Jan 1 (MAX_YEAR+1) (fail)
        assert!(
            find_next_date(
                first_day_of_year(NaiveDate::MAX.year())?,
                DateFreq::YearStart
            )
            .is_err()
        );

        // Test finding next year end after Dec 31 (MAX_YEAR-1) -> Dec 31 MAX_YEAR (ok)
        assert!(
            find_next_date(
                last_day_of_year(NaiveDate::MAX.year() - 1)?,
                DateFreq::YearEnd
            )
            .is_ok()
        );

        // Test finding next year end after Dec 31 MAX_YEAR -> Dec 31 (MAX_YEAR+1) (fail)
        assert!(
            find_next_date(last_day_of_year(NaiveDate::MAX.year())?, DateFreq::YearEnd).is_err()
        ); // Fails calculating MAX_YEAR+1

        Ok(())
    }

    // --- Tests for DatesGenerator ---

    #[test]
    fn test_generator_new_zero_periods() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 1, 1);
        let freq = DateFreq::Daily;
        let n_periods = 0;
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;
        assert_eq!(generator.next(), None); // Immediately exhausted
        Ok(())
    }

    #[test]
    fn test_generator_new_fail_find_first() -> Result<(), Box<dyn Error>> {
        let start_date = NaiveDate::MAX;
        // Use a frequency that requires finding the *next* day if MAX isn't the target.
        let freq = DateFreq::WeeklyMonday;
        let n_periods = 1;
        let result = DatesGenerator::new(start_date, freq, n_periods);
        // This fails if MAX is not a Monday, because find_first tries MAX.succ_opt()
        if NaiveDate::MAX.weekday() != Weekday::Mon {
            assert!(result.is_err());
        } else {
            // If MAX *is* a Monday, new() succeeds.
            assert!(result.is_ok());
        }
        Ok(())
    }

    #[test]
    fn test_generator_daily() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 10); // Fri
        let freq = DateFreq::Daily;
        let n_periods = 4;
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 10))); // Fri
        assert_eq!(generator.next(), Some(date(2023, 11, 11))); // Sat
        assert_eq!(generator.next(), Some(date(2023, 11, 12))); // Sun
        assert_eq!(generator.next(), Some(date(2023, 11, 13))); // Mon
        assert_eq!(generator.next(), None); // Exhausted

        // Test collecting
        let generator_collect = DatesGenerator::new(start_date, freq, n_periods)?;
        assert_eq!(
            generator_collect.collect::<Vec<_>>(),
            vec![
                date(2023, 11, 10),
                date(2023, 11, 11),
                date(2023, 11, 12),
                date(2023, 11, 13)
            ]
        );

        Ok(())
    }

    #[test]
    fn test_generator_weekly_monday() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 8); // Wed
        let freq = DateFreq::WeeklyMonday;
        let n_periods = 3;
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 13)));
        assert_eq!(generator.next(), Some(date(2023, 11, 20)));
        assert_eq!(generator.next(), Some(date(2023, 11, 27)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_weekly_friday() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 11); // Sat
        let freq = DateFreq::WeeklyFriday;
        let n_periods = 3;
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 17)));
        assert_eq!(generator.next(), Some(date(2023, 11, 24)));
        assert_eq!(generator.next(), Some(date(2023, 12, 1)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_month_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 10, 15); // Mid-Oct
        let freq = DateFreq::MonthStart;
        let n_periods = 4; // Nov, Dec, Jan, Feb
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 1)));
        assert_eq!(generator.next(), Some(date(2023, 12, 1)));
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2024, 2, 1)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_month_end_leap() -> Result<(), Box<dyn Error>> {
        let start_date = date(2024, 1, 31); // Jan 31
        let freq = DateFreq::MonthEnd;
        let n_periods = 3; // Jan, Feb (leap), Mar
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        // find_first for Jan 31 returns Jan 31
        assert_eq!(generator.next(), Some(date(2024, 1, 31)));
        // find_next finds Feb 29
        assert_eq!(generator.next(), Some(date(2024, 2, 29)));
        // find_next finds Mar 31
        assert_eq!(generator.next(), Some(date(2024, 3, 31)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_quarter_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 8, 1); // Mid-Q3
        let freq = DateFreq::QuarterStart;
        let n_periods = 3; // Q4'23, Q1'24, Q2'24
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 10, 1)));
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2024, 4, 1)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_quarter_end() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 1); // Mid-Q4
        let freq = DateFreq::QuarterEnd;
        let n_periods = 3; // Q4'23, Q1'24, Q2'24
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        // find_first for Nov 1 (Q4) returns Dec 31 (Q4 end)
        assert_eq!(generator.next(), Some(date(2023, 12, 31)));
        // find_next finds Mar 31 (Q1 end)
        assert_eq!(generator.next(), Some(date(2024, 3, 31)));
        // find_next finds Jun 30 (Q2 end)
        assert_eq!(generator.next(), Some(date(2024, 6, 30)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_year_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 1, 1); // Jan 1
        let freq = DateFreq::YearStart;
        let n_periods = 3; // 2023, 2024, 2025
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 1, 1)));
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2025, 1, 1)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_year_end() -> Result<(), Box<dyn Error>> {
        let start_date = date(2022, 12, 31); // Dec 31
        let freq = DateFreq::YearEnd;
        let n_periods = 3; // 2022, 2023, 2024
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        // find_first for Dec 31 '22 returns Dec 31 '22
        assert_eq!(generator.next(), Some(date(2022, 12, 31)));
        // find_next finds Dec 31 '23
        assert_eq!(generator.next(), Some(date(2023, 12, 31)));
        // find_next finds Dec 31 '24
        assert_eq!(generator.next(), Some(date(2024, 12, 31)));
        assert_eq!(generator.next(), None);
        Ok(())
    }

    #[test]
    fn test_generator_stops_after_error_finding_next() -> Result<(), Box<dyn Error>> {
        let start_year = NaiveDate::MAX.year();
        let start_date = last_day_of_year(start_year - 1)?; // Dec 31 of year before MAX
        let freq = DateFreq::YearEnd;
        let n_periods = 3; // Try for YE(MAX-1), YE(MAX), YE(MAX+1) - last should fail
        let mut generator = DatesGenerator::new(start_date, freq, n_periods)?;

        // find_first returns start_date (YE MAX-1)
        assert_eq!(generator.next(), Some(start_date));
        // find_next finds YE(MAX)
        assert_eq!(generator.next(), Some(last_day_of_year(start_year)?)); // Should be MAX
        // find_next tries YE(MAX+1) - this call to find_next_date fails internally
        assert_eq!(generator.next(), None); // Returns None because internal find_next_date failed

        // Check internal state after the call that returned None
        // When Some(YE MAX) was returned, periods_remaining became 1.
        // The next call enters the match, calls find_next_date (fails -> .ok() is None),
        // sets next_date_candidate=None, decrements periods_remaining to 0, returns Some(YE MAX).
        // --> NO, the code was: set candidate=find().ok(), THEN decrement.
        // Let's revisit Iterator::next logic:
        // 1. periods_remaining = 1, next_date_candidate = Some(YE MAX)
        // 2. Enter match arm
        // 3. find_next_date(YE MAX, YE) -> Err
        // 4. self.next_date_candidate = Err.ok() -> None
        // 5. self.periods_remaining -= 1 -> becomes 0
        // 6. return Some(YE MAX) <-- This was the bug in my reasoning. It returns the *current* date first.
        // State after returning Some(YE MAX): periods_remaining = 0, next_date_candidate = None
        // Next call to generator.next():
        // 1. periods_remaining = 0
        // 2. Enter the `_` arm of the match
        // 3. self.periods_remaining = 0 (no change)
        // 4. self.next_date_candidate = None (no change)
        // 5. return None

        // State after the *first* None is returned:
        assert_eq!(generator.periods_remaining, 0); // Corrected assertion
        assert!(generator.next_date_candidate.is_none());

        // Calling next() again should also return None
        assert_eq!(generator.next(), None);
        assert_eq!(generator.periods_remaining, 0);

        Ok(())
    }
} // end mod tests
