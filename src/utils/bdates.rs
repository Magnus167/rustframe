use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::result::Result;
use std::str::FromStr; // Import FromStr trait

/// Represents the frequency at which business dates should be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BDateFreq {
    Daily,
    WeeklyMonday,
    MonthStart,
    QuarterStart,
    YearStart,
    MonthEnd,
    QuarterEnd,
    WeeklyFriday,
    YearEnd,
}

/// Indicates whether the first or last date in a periodic group (like month, quarter)
/// is selected for the frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    Start, // Indicates picking the first valid business date in a group's period.
    End,   // Indicates picking the last valid business day in a group's period.
}

impl BDateFreq {
    /// Attempts to parse a frequency string into a `BDateFreq` enum.
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
        // Use the FromStr implementation directly
        freq.parse()
    }

    /// Returns the canonical string representation of the frequency.
    ///
    /// This returns the primary code (e.g., "D", "W", "Y", "YE"), not the aliases.
    pub fn to_string(&self) -> String {
        let r = match self {
            BDateFreq::Daily => "D",
            BDateFreq::WeeklyMonday => "W",
            BDateFreq::MonthStart => "M",
            BDateFreq::QuarterStart => "Q",
            BDateFreq::YearStart => "Y", // Changed to "Y"
            BDateFreq::MonthEnd => "ME",
            BDateFreq::QuarterEnd => "QE",
            BDateFreq::WeeklyFriday => "WF",
            BDateFreq::YearEnd => "YE",
        };
        r.to_string()
    }

    /// Determines whether the frequency represents a start-of-period or end-of-period aggregation.
    pub fn agg_type(&self) -> AggregationType {
        match self {
            BDateFreq::Daily
            | BDateFreq::WeeklyMonday
            | BDateFreq::MonthStart
            | BDateFreq::QuarterStart
            | BDateFreq::YearStart => AggregationType::Start,

            BDateFreq::WeeklyFriday
            | BDateFreq::MonthEnd
            | BDateFreq::QuarterEnd
            | BDateFreq::YearEnd => AggregationType::End,
        }
    }
}

// Implement FromStr for BDateFreq to allow parsing directly using `parse()`
impl FromStr for BDateFreq {
    type Err = Box<dyn Error>;

    /// Attempts to parse a frequency string slice into a `BDateFreq` enum.
    ///
    /// Supports various frequency codes and common aliases.
    ///
    /// | Code | Alias   | Description         |
    /// |------|---------|---------------------|
    /// | D    |         | Daily               |
    /// | W    | WS      | Weekly Monday       |
    /// | M    | MS      | Month Start         |
    /// | Q    | QS      | Quarter Start       |
    /// | Y    | A, AS, YS | Year Start        |
    /// | ME   |         | Month End           |
    /// | QE   |         | Quarter End         |
    /// | WF   |         | Weekly Friday       |
    /// | YE   | AE      | Year End (Annual)   |
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
            "D" => BDateFreq::Daily,
            "W" | "WS" => BDateFreq::WeeklyMonday,
            "M" | "MS" => BDateFreq::MonthStart,
            "Q" | "QS" => BDateFreq::QuarterStart,
            "Y" | "A" | "AS" | "YS" => BDateFreq::YearStart, // Added Y, YS, A, AS aliases
            "ME" => BDateFreq::MonthEnd,
            "QE" => BDateFreq::QuarterEnd,
            "WF" => BDateFreq::WeeklyFriday,
            "YE" | "AE" => BDateFreq::YearEnd, // Added AE alias
            _ => return Err(format!("Invalid frequency specified: {}", freq).into()),
        };
        Ok(r)
    }
}

/// Represents a list of business dates generated between a start and end date
/// at a specified frequency. Provides methods to retrieve the full list,
/// count, or dates grouped by period.
#[derive(Debug, Clone)]
pub struct BDatesList {
    start_date_str: String,
    end_date_str: String,
    freq: BDateFreq,
    // Optional: Cache the generated list to avoid re-computation?
    // For now, we recompute each time list(), count(), or groups() is called.
    // cached_list: Option<Vec<NaiveDate>>,
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

/// Represents a collection of business dates generated according to specific rules.
///
/// It can be defined either by a start and end date range or by a start date
/// and a fixed number of periods. It provides methods to retrieve the dates
/// as a flat list, count them, or group them by their natural period
/// (e.g., month, quarter).
///
/// Business days are typically Monday to Friday. Weekend dates are skipped or
/// adjusted depending on the frequency rules.
///
/// # Examples
///
/// **1. Using `new` (Start and End Date):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use bdates::{BDatesList, BDateFreq}; // Replace bdates with your actual crate/module name
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2023-11-01".to_string(); // Wednesday
/// let end_date = "2023-11-07".to_string();   // Tuesday
/// let freq = BDateFreq::Daily;
///
/// let bdates = BDatesList::new(start_date, end_date, freq);
///
/// let expected_dates = vec![
///     NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(), // Wed
///     NaiveDate::from_ymd_opt(2023, 11, 2).unwrap(), // Thu
///     NaiveDate::from_ymd_opt(2023, 11, 3).unwrap(), // Fri
///     NaiveDate::from_ymd_opt(2023, 11, 6).unwrap(), // Mon
///     NaiveDate::from_ymd_opt(2023, 11, 7).unwrap(), // Tue
/// ];
///
/// assert_eq!(bdates.list()?, expected_dates);
/// assert_eq!(bdates.count()?, 5);
/// # Ok(())
/// # }
/// ```
///
/// **2. Using `from_n_periods` (Start Date and Count):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use bdates::{BDatesList, BDateFreq}; // Replace bdates with your actual crate/module name
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2024-02-28".to_string(); // Wednesday
/// let freq = BDateFreq::WeeklyFriday;
/// let n_periods = 3;
///
/// let bdates = BDatesList::from_n_periods(start_date, freq, n_periods)?;
///
/// // The first Friday on or after 2024-02-28 is Mar 1.
/// // The next two Fridays are Mar 8 and Mar 15.
/// let expected_dates = vec![
///     NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
///     NaiveDate::from_ymd_opt(2024, 3, 8).unwrap(),
///     NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
/// ];
///
/// assert_eq!(bdates.list()?, expected_dates);
/// assert_eq!(bdates.count()?, 3);
/// assert_eq!(bdates.start_date_str(), "2024-02-28"); // Keeps original start string
/// assert_eq!(bdates.end_date_str(), "2024-03-15");   // End date is the last generated date
/// # Ok(())
/// # }
/// ```
///
/// **3. Using `groups()`:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use bdates::{BDatesList, BDateFreq}; // Replace bdates with your actual crate/module name
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start_date = "2023-11-20".to_string(); // Mon, Week 47
/// let end_date = "2023-12-08".to_string();   // Fri, Week 49
/// let freq = BDateFreq::WeeklyMonday;
///
/// let bdates = BDatesList::new(start_date, end_date, freq);
///
/// // Mondays in range: Nov 20, Nov 27, Dec 4
/// let groups = bdates.groups()?;
///
/// assert_eq!(groups.len(), 3); // One group per week containing a Monday
/// assert_eq!(groups[0], vec![NaiveDate::from_ymd_opt(2023, 11, 20).unwrap()]); // Week 47
/// assert_eq!(groups[1], vec![NaiveDate::from_ymd_opt(2023, 11, 27).unwrap()]); // Week 48
/// assert_eq!(groups[2], vec![NaiveDate::from_ymd_opt(2023, 12, 4).unwrap()]);  // Week 49
/// # Ok(())
/// # }
/// ```
impl BDatesList {
    /// Creates a new `BDatesList` instance defined by a start and end date.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The inclusive start date as a string (e.g., "YYYY-MM-DD").
    /// * `end_date_str` - The inclusive end date as a string (e.g., "YYYY-MM-DD").
    /// * `freq` - The frequency for generating dates.
    pub fn new(start_date_str: String, end_date_str: String, freq: BDateFreq) -> Self {
        BDatesList {
            start_date_str,
            end_date_str,
            freq,
        }
    }

    /// Creates a new `BDatesList` instance defined by a start date, frequency,
    /// and the number of periods (dates) to generate.
    ///
    /// This calculates the required dates using a `BDatesGenerator` and determines
    /// the effective end date based on the last generated date.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The start date as a string (e.g., "YYYY-MM-DD"). The first generated date will be on or after this date.
    /// * `freq` - The frequency for generating dates.
    /// * `n_periods` - The exact number of business dates to generate according to the frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `start_date_str` cannot be parsed.
    /// * `n_periods` is 0 (as this would result in an empty list and no defined end date).
    pub fn from_n_periods(
        start_date_str: String,
        freq: BDateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if n_periods == 0 {
            return Err("n_periods must be greater than 0".into());
        }

        let start_date = NaiveDate::parse_from_str(&start_date_str, "%Y-%m-%d")?;

        // Use the generator to find all the dates
        let generator = BDatesGenerator::new(start_date, freq, n_periods)?;
        let dates: Vec<NaiveDate> = generator.collect();

        // Should always have at least one date if n_periods > 0 and generator construction succeeded
        let last_date = dates
            .last()
            .ok_or("Generator failed to produce dates even though n_periods > 0")?;

        let end_date_str = last_date.format("%Y-%m-%d").to_string();

        Ok(BDatesList {
            start_date_str, // Keep the original start date string
            end_date_str,
            freq,
        })
    }

    /// Returns the flat list of business dates within the specified range and frequency.
    ///
    /// The list is guaranteed to be sorted chronologically.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn list(&self) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
        // Delegate the core logic to the internal helper function
        get_bdates_list_with_freq(&self.start_date_str, &self.end_date_str, self.freq)
    }

    /// Returns the count of business dates within the specified range and frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed (as it
    /// calls `list` internally).
    pub fn count(&self) -> Result<usize, Box<dyn Error>> {
        // Get the list and return its length. Uses map to handle the Result elegantly.
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
        // Get the sorted list of all dates first. This sorted order is crucial
        // for ensuring the inner vectors (dates within groups) are also sorted
        // as we insert into the HashMap.
        let dates = self.list()?;

        // Use a HashMap to collect dates into their respective groups.
        let mut groups: HashMap<GroupKey, Vec<NaiveDate>> = HashMap::new();

        for date in dates {
            // Determine the grouping key based on frequency.
            let key = match self.freq {
                BDateFreq::Daily => GroupKey::Daily(date),
                BDateFreq::WeeklyMonday | BDateFreq::WeeklyFriday => {
                    // Use ISO week number for consistent weekly grouping across year boundaries
                    let iso_week = date.iso_week();
                    GroupKey::Weekly(iso_week.year(), iso_week.week())
                }
                BDateFreq::MonthStart | BDateFreq::MonthEnd => {
                    GroupKey::Monthly(date.year(), date.month())
                }
                BDateFreq::QuarterStart | BDateFreq::QuarterEnd => {
                    GroupKey::Quarterly(date.year(), month_to_quarter(date.month()))
                }
                BDateFreq::YearStart | BDateFreq::YearEnd => GroupKey::Yearly(date.year()),
            };

            // Add the current date to the vector corresponding to the determined key.
            // entry().or_insert_with() gets a mutable reference to the vector for the key,
            // inserting a new empty vector if the key doesn't exist yet.
            groups.entry(key).or_insert_with(Vec::new).push(date); // Using or_insert_with is slightly more idiomatic
        }

        // Convert the HashMap into a vector of (key, vector_of_dates) tuples.
        let mut sorted_groups: Vec<(GroupKey, Vec<NaiveDate>)> = groups.into_iter().collect();

        // Sort the vector of groups by the `GroupKey`. Since `GroupKey` derives `Ord`,
        // this sorts the groups chronologically (Yearly < Quarterly < Monthly < Weekly < Daily,
        // then by year, quarter, month, week, or date within each category).
        sorted_groups.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        // The dates *within* each group (`Vec<NaiveDate>`) are already sorted
        // because they were pushed in the order they appeared in the initially
        // sorted `dates` vector obtained from `self.list()`.
        // If the source `dates` wasn't guaranteed sorted, or for clarity,
        // an inner sort could be added here:
        // for (_, dates_in_group) in sorted_groups.iter_mut() {
        //     dates_in_group.sort();
        // }

        // Extract just the vectors of dates from the sorted tuples, discarding the keys.
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
    pub fn freq(&self) -> BDateFreq {
        self.freq
    }

    /// Returns the canonical string representation of the frequency.
    pub fn freq_str(&self) -> String {
        self.freq.to_string()
    }
}

// --- Business Date Generator (Iterator) ---

/// An iterator that generates a sequence of business dates based on a start date,
/// frequency, and a specified number of periods.
///
/// This implements the `Iterator` trait, allowing generation of dates one by one.
/// It's useful when you need to process dates lazily or only need a fixed number
/// starting from a specific point, without necessarily defining an end date beforehand.
///
/// # Examples
///
/// **1. Basic Iteration:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use bdates::{BDatesGenerator, BDateFreq}; // Replace bdates with your actual crate/module name
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start = NaiveDate::from_ymd_opt(2023, 12, 28).unwrap(); // Thursday
/// let freq = BDateFreq::MonthEnd;
/// let n_periods = 4; // Dec '23, Jan '24, Feb '24, Mar '24
///
/// let mut generator = BDatesGenerator::new(start, freq, n_periods)?;
///
/// // First month-end on or after 2023-12-28 is 2023-12-29
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2023, 12, 29).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 31).unwrap()));
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 2, 29).unwrap())); // Leap year
/// assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 3, 29).unwrap())); // Mar 31 is Sun
/// assert_eq!(generator.next(), None); // Exhausted
/// # Ok(())
/// # }
/// ```
///
/// **2. Collecting into a Vec:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// # use bdates::{BDatesGenerator, BDateFreq}; // Replace bdates with your actual crate/module name
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let start = NaiveDate::from_ymd_opt(2024, 4, 29).unwrap(); // Monday
/// let freq = BDateFreq::Daily;
/// let n_periods = 5;
///
/// let generator = BDatesGenerator::new(start, freq, n_periods)?;
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
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BDatesGenerator {
    freq: BDateFreq,
    periods_remaining: usize,
    // Stores the *next* date to be yielded by the iterator.
    // This is None initially or when the iterator is exhausted.
    next_date_candidate: Option<NaiveDate>,
}

impl BDatesGenerator {
    /// Creates a new `BDatesGenerator`.
    ///
    /// It calculates the first valid business date based on the `start_date` and `freq`,
    /// which will be the first item yielded by the iterator.
    ///
    /// # Arguments
    ///
    /// * `start_date` - The date from which to start searching for the first valid business date.
    /// * `freq` - The frequency for generating dates.
    /// * `n_periods` - The total number of business dates to generate.
    ///
    /// # Errors
    ///
    /// Can potentially return an error if date calculations lead to overflows,
    /// though this is highly unlikely with realistic date ranges. (Currently returns Ok).
    /// Note: The internal `find_first_bdate_on_or_after` might panic on extreme date overflows,
    /// but practical usage should be safe.
    pub fn new(
        start_date: NaiveDate,
        freq: BDateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let first_date = if n_periods > 0 {
            Some(find_first_bdate_on_or_after(start_date, freq))
        } else {
            None // No dates to generate if n_periods is 0
        };

        Ok(BDatesGenerator {
            freq,
            periods_remaining: n_periods,
            next_date_candidate: first_date,
        })
    }
}

impl Iterator for BDatesGenerator {
    type Item = NaiveDate;

    /// Returns the next business date in the sequence, or `None` if `n_periods`
    /// dates have already been generated.
    fn next(&mut self) -> Option<Self::Item> {
        // Check if exhausted or if there was no initial date
        if self.periods_remaining == 0 || self.next_date_candidate.is_none() {
            return None;
        }

        // Get the date to return (unwrap is safe due to the check above)
        let current_date = self.next_date_candidate.unwrap();

        // Prepare the *next* candidate for the subsequent call
        self.next_date_candidate = Some(find_next_bdate(current_date, self.freq));

        // Decrement the count
        self.periods_remaining -= 1;

        // Return the stored current date
        Some(current_date)
    }
}

// --- Internal helper functions (not part of the public API) ---

/// Generates the flat list of business dates for the given range and frequency.
///
/// Filters out weekends and ensures the final list is sorted. This is the core
/// generation logic used by `BDatesList::list` and `BDatesList::groups`.
///
/// # Arguments (Internal)
///
/// * `start_date_str` - Inclusive start date string.
/// * `end_date_str` - Inclusive end date string.
/// * `freq` - The frequency.
///
/// # Errors (Internal)
///
/// Returns an error if date strings are invalid.
fn get_bdates_list_with_freq(
    start_date_str: &str,
    end_date_str: &str,
    freq: BDateFreq,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    // Parse the start and end dates, returning error if parsing fails.
    let start_date = NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d")?;
    let end_date = NaiveDate::parse_from_str(end_date_str, "%Y-%m-%d")?;

    // Handle edge case where end date is before start date.
    if start_date > end_date {
        return Ok(Vec::new());
    }

    // Collect dates based on the specified frequency.
    let mut dates = match freq {
        BDateFreq::Daily => collect_daily(start_date, end_date),
        BDateFreq::WeeklyMonday => collect_weekly(start_date, end_date, Weekday::Mon),
        BDateFreq::WeeklyFriday => collect_weekly(start_date, end_date, Weekday::Fri),
        BDateFreq::MonthStart => {
            collect_monthly(start_date, end_date, /*want_first_day=*/ true)
        }
        BDateFreq::MonthEnd => {
            collect_monthly(start_date, end_date, /*want_first_day=*/ false)
        }
        BDateFreq::QuarterStart => {
            collect_quarterly(start_date, end_date, /*want_first_day=*/ true)
        }
        BDateFreq::QuarterEnd => {
            collect_quarterly(start_date, end_date, /*want_first_day=*/ false)
        }
        BDateFreq::YearStart => collect_yearly(start_date, end_date, /*want_first_day=*/ true),
        BDateFreq::YearEnd => collect_yearly(start_date, end_date, /*want_first_day=*/ false),
    };

    // Filter out any weekend days. While the core logic aims for business days,
    // this ensures robustness against edge cases where computed dates might fall
    // on a weekend (e.g., first day of month being Saturday).
    // Note: This retain is redundant if collect_* functions are correct, but adds safety.
    // It's essential for Daily, less so for others if they always return bdays.
    dates.retain(|d| is_weekday(*d));

    // Ensure the final list is sorted. The `collect_*` functions generally
    // produce sorted output, but an explicit sort guarantees it.
    dates.sort();

    Ok(dates)
}

/* ---------------------- Low-Level Date Collection Functions (Internal) ---------------------- */
// These functions generate dates within a *range* [start_date, end_date]

/// Returns all business days (Mon-Fri) day-by-day within the range.
fn collect_daily(start_date: NaiveDate, end_date: NaiveDate) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut current = start_date;
    while current <= end_date {
        if is_weekday(current) {
            result.push(current);
        }
        // Use succ_opt() and expect(), assuming valid date range and no overflow in practical scenarios
        current = current
            .succ_opt()
            .expect("date overflow near end of supported range");
    }
    result
}

/// Returns the specified `target_weekday` in each week within the range.
fn collect_weekly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    target_weekday: Weekday,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();

    // Find the first target_weekday on or after the start date.
    let mut current = move_to_weekday_on_or_after(start_date, target_weekday);

    // Step through the range in 7-day increments.
    while current <= end_date {
        // Ensure the found date is actually a weekday (should be Mon/Fri but belt-and-suspenders)
        if is_weekday(current) {
            result.push(current);
        }
        // Use checked_add_signed for safety, though basic addition is likely fine for 7 days.
        current = current
            .checked_add_signed(Duration::days(7))
            .expect("date overflow adding 7 days");
    }
    result
}

/// Returns either the first or last business day in each month of the range.
fn collect_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();

    let mut year = start_date.year();
    let mut month = start_date.month();

    // Helper closure to advance year and month by one month.
    let next_month =
        |(yr, mo): (i32, u32)| -> (i32, u32) { if mo == 12 { (yr + 1, 1) } else { (yr, mo + 1) } };

    // Iterate month by month from the start date's month up to or past the end date's month.
    loop {
        // Compute the candidate date (first or last business day) for the current month.
        // Use _opt and expect(), expecting valid month/year combinations within realistic ranges.
        let candidate = if want_first_day {
            first_business_day_of_month(year, month)
        } else {
            last_business_day_of_month(year, month)
        };

        // If the candidate is after the end date, we've gone past the range, so stop.
        if candidate > end_date {
            break;
        }

        // If the candidate is within the specified range [start_date, end_date], add it.
        if candidate >= start_date {
            // Ensure it's actually a weekday (should be, but adds safety)
            if is_weekday(candidate) {
                result.push(candidate);
            }
        }
        // Note: We don't break if candidate < start_date because a later month's candidate
        // might be within the range.

        // Check if the current month is the last month we should process
        if year > end_date.year() || (year == end_date.year() && month >= end_date.month()) {
            // If we just processed the end_date's month, stop.
            // Need >= because we need to include the end date's month itself if its candidate is valid.
            break;
        }

        // Advance to the next month.
        let (ny, nm) = next_month((year, month));
        year = ny;
        month = nm;

        // Safety break: Stop if we have moved clearly past the end date's year.
        // This check is technically redundant given the loop condition above, but harmless.
        if year > end_date.year() + 1 {
            break; // Avoid potential infinite loops in unexpected scenarios
        }
    }

    result
}

/// Return either the first or last business day in each quarter of the range.
fn collect_quarterly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();

    let mut year = start_date.year();
    // Start from the quarter containing the start date.
    let mut q = month_to_quarter(start_date.month());

    // Iterate quarter by quarter until we pass the end date.
    loop {
        // Compute the candidate date (first or last business day) for the current quarter.
        // Use _opt and expect(), expecting valid quarter/year combinations.
        let candidate = if want_first_day {
            first_business_day_of_quarter(year, q)
        } else {
            last_business_day_of_quarter(year, q)
        };

        // If the candidate is after the end date, we've gone past the range, so stop.
        if candidate > end_date {
            break;
        }

        // If the candidate is within the specified range [start_date, end_date], add it.
        if candidate >= start_date {
            // Ensure it's actually a weekday (should be, but adds safety)
            if is_weekday(candidate) {
                result.push(candidate);
            }
        }
        // Note: We don't break if candidate < start_date because a later quarter
        // might be within the range.

        // Check if the current quarter is the last one we should process
        let end_q = month_to_quarter(end_date.month());
        if year > end_date.year() || (year == end_date.year() && q >= end_q) {
            break; // Stop after processing the end_date's quarter
        }

        // Advance to the next quarter.
        if q == 4 {
            year += 1;
            q = 1;
        } else {
            q += 1;
        }

        // Safety break
        if year > end_date.year() + 1 {
            break;
        }
    }

    result
}

/// Return either the first or last business day in each year of the range.
fn collect_yearly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    // Start from the year of the start date.
    let mut year = start_date.year();

    // Iterate year by year until we pass the end date's year.
    while year <= end_date.year() {
        // Compute the candidate date (first or last business day) for the current year.
        // Use _opt and expect(), expecting valid year.
        let candidate = if want_first_day {
            first_business_day_of_year(year)
        } else {
            last_business_day_of_year(year)
        };

        // If the candidate is within the specified range [start_date, end_date], add it.
        if candidate >= start_date && candidate <= end_date {
            // Ensure it's actually a weekday (should be, but adds safety)
            if is_weekday(candidate) {
                result.push(candidate);
            }
        } else if want_first_day && candidate > end_date {
            // Optimization: If the *first* bday of the year is already past end_date,
            // no subsequent year's first bday will be in range.
            // Similar logic applies for last bday if candidate > end_date, but it's less likely to trigger early.
            break;
        }
        // Note: We don't break if candidate < start_date because a later year's candidate
        // might be within the range (e.g. start_date 2023-12-15, YE freq, candidate for 2023 is 2023-12-29 (ok),
        // candidate for 2024 is 2024-12-31 (could be ok)).

        year += 1;
    }
    result
}

/* ---------------------- Core Date Utility Functions (Internal) ---------------------- */

/// Checks if a given date is a weekday (Monday-Friday).
fn is_weekday(date: NaiveDate) -> bool {
    !matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
}

/// Given a date and a `target_weekday`, returns the date that is the first
/// `target_weekday` on or after the given date.
fn move_to_weekday_on_or_after(date: NaiveDate, target: Weekday) -> NaiveDate {
    let mut current = date;
    while current.weekday() != target {
        // Use succ_opt() and expect(), assuming valid date and no overflow
        current = current
            .succ_opt()
            .expect("date overflow moving to next weekday");
    }
    current
}

/// Return the earliest business day of the given (year, month).
fn first_business_day_of_month(year: i32, month: u32) -> NaiveDate {
    // Start with the 1st of the month. Use _opt and expect(), assuming valid Y/M.
    let mut d = NaiveDate::from_ymd_opt(year, month, 1).expect("invalid year-month combination");
    // If it’s Sat/Sun, move forward until we find a weekday.
    while !is_weekday(d) {
        // Use succ_opt() and expect(), assuming valid date and no overflow.
        d = d.succ_opt().expect("date overflow finding first bday");
    }
    d
}

/// Return the latest business day of the given (year, month).
fn last_business_day_of_month(year: i32, month: u32) -> NaiveDate {
    let last_dom = days_in_month(year, month);
    // Use _opt and expect(), assuming valid Y/M/D combination.
    let mut d =
        NaiveDate::from_ymd_opt(year, month, last_dom).expect("invalid year-month-day combination");
    // If it’s Sat/Sun, move backward until we find a weekday.
    while !is_weekday(d) {
        // Use pred_opt() and expect(), assuming valid date and no underflow.
        d = d.pred_opt().expect("date underflow finding last bday");
    }
    d
}

/// Returns the number of days in a given month and year.
fn days_in_month(year: i32, month: u32) -> u32 {
    // A common trick: find the first day of the *next* month, then subtract one day.
    // This correctly handles leap years.
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    // Use _opt and expect(), assuming valid Y/M combination (start of next month).
    let first_of_next = NaiveDate::from_ymd_opt(ny, nm, 1).expect("invalid next year-month");
    // Use pred_opt() and expect(), assuming valid date and no underflow (first of month - 1).
    let last_of_this = first_of_next
        .pred_opt()
        .expect("invalid date before first of month");
    last_of_this.day()
}

/// Converts a month number (1-12) to a quarter number (1-4).
fn month_to_quarter(m: u32) -> u32 {
    match m {
        1..=3 => 1,
        4..=6 => 2,
        7..=9 => 3,
        10..=12 => 4,
        _ => panic!("Invalid month: {}", m), // Should not happen with valid dates
    }
}

/// Returns the 1st day of the month that starts a given (year, quarter).
fn quarter_start_month(quarter: u32) -> u32 {
    match quarter {
        1 => 1,
        2 => 4,
        3 => 7,
        4 => 10,
        _ => panic!("invalid quarter: {}", quarter), // This function expects quarter 1-4
    }
}

/// Return the earliest business day in the given (year, quarter).
fn first_business_day_of_quarter(year: i32, quarter: u32) -> NaiveDate {
    let month = quarter_start_month(quarter);
    first_business_day_of_month(year, month)
}

/// Return the last business day in the given (year, quarter).
fn last_business_day_of_quarter(year: i32, quarter: u32) -> NaiveDate {
    // The last month of a quarter is the start month + 2.
    let last_month_in_quarter = match quarter {
        1 => 3,
        2 => 6,
        3 => 9,
        4 => 12,
        _ => panic!("invalid quarter: {}", quarter),
    };
    last_business_day_of_month(year, last_month_in_quarter)
}

/// Returns the earliest business day (Mon-Fri) of the given year.
fn first_business_day_of_year(year: i32) -> NaiveDate {
    // Start with Jan 1st. Use _opt and expect(), assuming valid Y/M/D combination.
    let mut d = NaiveDate::from_ymd_opt(year, 1, 1).expect("invalid year for Jan 1st");
    // If Jan 1st is a weekend, move forward to the next weekday.
    while !is_weekday(d) {
        // Use succ_opt() and expect(), assuming valid date and no overflow.
        d = d
            .succ_opt()
            .expect("date overflow finding first bday of year");
    }
    d
}

/// Returns the last business day (Mon-Fri) of the given year.
fn last_business_day_of_year(year: i32) -> NaiveDate {
    // Start with Dec 31st. Use _opt and expect(), assuming valid Y/M/D combination.
    let mut d = NaiveDate::from_ymd_opt(year, 12, 31).expect("invalid year for Dec 31st");
    // If Dec 31st is a weekend, move backward to the previous weekday.
    while !is_weekday(d) {
        // Use pred_opt() and expect(), assuming valid date and no underflow.
        d = d
            .pred_opt()
            .expect("date underflow finding last bday of year");
    }
    d
}

// --- Generator Helper Functions ---

/// Finds the *first* valid business date according to the frequency,
/// starting the search *on or after* the given `start_date`.
/// Panics on date overflow/underflow in extreme cases, but generally safe.
fn find_first_bdate_on_or_after(start_date: NaiveDate, freq: BDateFreq) -> NaiveDate {
    match freq {
        BDateFreq::Daily => {
            let mut d = start_date;
            while !is_weekday(d) {
                d = d
                    .succ_opt()
                    .expect("Date overflow finding first daily date");
            }
            d
        }
        BDateFreq::WeeklyMonday => move_to_weekday_on_or_after(start_date, Weekday::Mon),
        BDateFreq::WeeklyFriday => move_to_weekday_on_or_after(start_date, Weekday::Fri),
        BDateFreq::MonthStart => {
            let mut candidate = first_business_day_of_month(start_date.year(), start_date.month());
            if candidate < start_date {
                // If the first bday of the current month is before start_date,
                // we need the first bday of the *next* month.
                let (next_y, next_m) = if start_date.month() == 12 {
                    (start_date.year() + 1, 1)
                } else {
                    (start_date.year(), start_date.month() + 1)
                };
                candidate = first_business_day_of_month(next_y, next_m);
            }
            candidate
        }
        BDateFreq::MonthEnd => {
            let mut candidate = last_business_day_of_month(start_date.year(), start_date.month());
            if candidate < start_date {
                // If the last bday of current month is before start_date,
                // we need the last bday of the *next* month.
                let (next_y, next_m) = if start_date.month() == 12 {
                    (start_date.year() + 1, 1)
                } else {
                    (start_date.year(), start_date.month() + 1)
                };
                candidate = last_business_day_of_month(next_y, next_m);
            }
            candidate
        }
        BDateFreq::QuarterStart => {
            let current_q = month_to_quarter(start_date.month());
            let mut candidate = first_business_day_of_quarter(start_date.year(), current_q);
            if candidate < start_date {
                // If the first bday of the current quarter is before start_date,
                // we need the first bday of the *next* quarter.
                let (next_y, next_q) = if current_q == 4 {
                    (start_date.year() + 1, 1)
                } else {
                    (start_date.year(), current_q + 1)
                };
                candidate = first_business_day_of_quarter(next_y, next_q);
            }
            candidate
        }
        BDateFreq::QuarterEnd => {
            let current_q = month_to_quarter(start_date.month());
            let mut candidate = last_business_day_of_quarter(start_date.year(), current_q);
            if candidate < start_date {
                // If the last bday of the current quarter is before start_date,
                // we need the last bday of the *next* quarter.
                let (next_y, next_q) = if current_q == 4 {
                    (start_date.year() + 1, 1)
                } else {
                    (start_date.year(), current_q + 1)
                };
                candidate = last_business_day_of_quarter(next_y, next_q);
            }
            candidate
        }
        BDateFreq::YearStart => {
            let mut candidate = first_business_day_of_year(start_date.year());
            if candidate < start_date {
                // If the first bday of the current year is before start_date,
                // we need the first bday of the *next* year.
                candidate = first_business_day_of_year(start_date.year() + 1);
            }
            candidate
        }
        BDateFreq::YearEnd => {
            let mut candidate = last_business_day_of_year(start_date.year());
            if candidate < start_date {
                // If the last bday of the current year is before start_date,
                // we need the last bday of the *next* year.
                candidate = last_business_day_of_year(start_date.year() + 1);
            }
            candidate
        }
    }
}

/// Finds the *next* valid business date according to the frequency,
/// given the `current_date` (which is assumed to be a valid date previously generated).
/// Panics on date overflow/underflow in extreme cases, but generally safe.
fn find_next_bdate(current_date: NaiveDate, freq: BDateFreq) -> NaiveDate {
    match freq {
        BDateFreq::Daily => {
            let mut next_day = current_date
                .succ_opt()
                .expect("Date overflow finding next daily");
            while !is_weekday(next_day) {
                next_day = next_day
                    .succ_opt()
                    .expect("Date overflow finding next daily weekday");
            }
            next_day
        }
        BDateFreq::WeeklyMonday | BDateFreq::WeeklyFriday => {
            // Assuming current_date is already a Mon/Fri, the next one is 7 days later.
            current_date
                .checked_add_signed(Duration::days(7))
                .expect("Date overflow adding 7 days")
        }
        BDateFreq::MonthStart => {
            let (next_y, next_m) = if current_date.month() == 12 {
                (current_date.year() + 1, 1)
            } else {
                (current_date.year(), current_date.month() + 1)
            };
            first_business_day_of_month(next_y, next_m)
        }
        BDateFreq::MonthEnd => {
            let (next_y, next_m) = if current_date.month() == 12 {
                (current_date.year() + 1, 1)
            } else {
                (current_date.year(), current_date.month() + 1)
            };
            last_business_day_of_month(next_y, next_m)
        }
        BDateFreq::QuarterStart => {
            let current_q = month_to_quarter(current_date.month());
            let (next_y, next_q) = if current_q == 4 {
                (current_date.year() + 1, 1)
            } else {
                (current_date.year(), current_q + 1)
            };
            first_business_day_of_quarter(next_y, next_q)
        }
        BDateFreq::QuarterEnd => {
            let current_q = month_to_quarter(current_date.month());
            let (next_y, next_q) = if current_q == 4 {
                (current_date.year() + 1, 1)
            } else {
                (current_date.year(), current_q + 1)
            };
            last_business_day_of_quarter(next_y, next_q)
        }
        BDateFreq::YearStart => first_business_day_of_year(current_date.year() + 1),
        BDateFreq::YearEnd => last_business_day_of_year(current_date.year() + 1),
    }
}

// --- Example Usage and Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    // Helper to create a NaiveDate for tests, handling the expect for fixed dates.
    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("Invalid date in test setup")
    }

    // --- BDateFreq Tests ---

    #[test]
    fn test_bdatefreq_from_str() -> Result<(), Box<dyn Error>> {
        assert_eq!(BDateFreq::from_str("D")?, BDateFreq::Daily);
        assert_eq!("D".parse::<BDateFreq>()?, BDateFreq::Daily); // Test FromStr impl
        assert_eq!(BDateFreq::from_str("W")?, BDateFreq::WeeklyMonday);
        assert_eq!(BDateFreq::from_str("M")?, BDateFreq::MonthStart);
        assert_eq!(BDateFreq::from_str("Q")?, BDateFreq::QuarterStart);

        // Test YearStart codes and aliases (Y, A, AS, YS)
        assert_eq!(BDateFreq::from_str("Y")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("A")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("AS")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("YS")?, BDateFreq::YearStart);
        assert_eq!("Y".parse::<BDateFreq>()?, BDateFreq::YearStart); // Test FromStr impl

        assert_eq!(BDateFreq::from_str("ME")?, BDateFreq::MonthEnd);
        assert_eq!(BDateFreq::from_str("QE")?, BDateFreq::QuarterEnd);
        assert_eq!(BDateFreq::from_str("WF")?, BDateFreq::WeeklyFriday);
        assert_eq!("WF".parse::<BDateFreq>()?, BDateFreq::WeeklyFriday); // Test FromStr impl

        // Test YearEnd codes and aliases (YE, AE)
        assert_eq!(BDateFreq::from_str("YE")?, BDateFreq::YearEnd);
        assert_eq!(BDateFreq::from_str("AE")?, BDateFreq::YearEnd);

        // Test aliases for other frequencies
        assert_eq!(BDateFreq::from_str("WS")?, BDateFreq::WeeklyMonday);
        assert_eq!(BDateFreq::from_str("MS")?, BDateFreq::MonthStart);
        assert_eq!(BDateFreq::from_str("QS")?, BDateFreq::QuarterStart);

        // Test invalid string
        assert!(BDateFreq::from_str("INVALID").is_err());
        assert!("INVALID".parse::<BDateFreq>().is_err()); // Test FromStr impl
        let err = BDateFreq::from_str("INVALID").unwrap_err();
        assert_eq!(err.to_string(), "Invalid frequency specified: INVALID");

        Ok(())
    }

    #[test]
    fn test_bdatefreq_to_string() {
        assert_eq!(BDateFreq::Daily.to_string(), "D");
        assert_eq!(BDateFreq::WeeklyMonday.to_string(), "W");
        assert_eq!(BDateFreq::MonthStart.to_string(), "M");
        assert_eq!(BDateFreq::QuarterStart.to_string(), "Q");
        assert_eq!(BDateFreq::YearStart.to_string(), "Y"); // Assert "Y"
        assert_eq!(BDateFreq::MonthEnd.to_string(), "ME");
        assert_eq!(BDateFreq::QuarterEnd.to_string(), "QE");
        assert_eq!(BDateFreq::WeeklyFriday.to_string(), "WF");
        assert_eq!(BDateFreq::YearEnd.to_string(), "YE");
    }

    #[test]
    fn test_bdatefreq_from_string() -> Result<(), Box<dyn Error>> {
        assert_eq!(BDateFreq::from_string("D".to_string())?, BDateFreq::Daily);
        assert!(BDateFreq::from_string("INVALID".to_string()).is_err());
        Ok(())
    }

    #[test]
    fn test_bdatefreq_agg_type() {
        assert_eq!(BDateFreq::Daily.agg_type(), AggregationType::Start);
        assert_eq!(BDateFreq::WeeklyMonday.agg_type(), AggregationType::Start);
        assert_eq!(BDateFreq::MonthStart.agg_type(), AggregationType::Start);
        assert_eq!(BDateFreq::QuarterStart.agg_type(), AggregationType::Start);
        assert_eq!(BDateFreq::YearStart.agg_type(), AggregationType::Start);

        assert_eq!(BDateFreq::WeeklyFriday.agg_type(), AggregationType::End);
        assert_eq!(BDateFreq::MonthEnd.agg_type(), AggregationType::End);
        assert_eq!(BDateFreq::QuarterEnd.agg_type(), AggregationType::End);
        assert_eq!(BDateFreq::YearEnd.agg_type(), AggregationType::End);
    }

    // --- BDatesList Property Tests ---

    #[test]
    fn test_bdates_list_properties_new() -> Result<(), Box<dyn Error>> {
        let start_str = "2023-01-01".to_string();
        let end_str = "2023-12-31".to_string();
        let freq = BDateFreq::QuarterEnd;
        let dates_list = BDatesList::new(start_str.clone(), end_str.clone(), freq);

        // check start_date_str
        assert_eq!(dates_list.start_date_str(), start_str);
        // check end_date_str
        assert_eq!(dates_list.end_date_str(), end_str);
        // check frequency enum
        assert_eq!(dates_list.freq(), freq);
        // check frequency string
        assert_eq!(dates_list.freq_str(), "QE");

        // Check parsed dates
        assert_eq!(dates_list.start_date()?, date(2023, 1, 1));
        assert_eq!(dates_list.end_date()?, date(2023, 12, 31));

        Ok(())
    }

    #[test]
    fn test_bdates_list_properties_from_n_periods() -> Result<(), Box<dyn Error>> {
        let start_str = "2023-01-01".to_string(); // Sunday
        let freq = BDateFreq::Daily;
        let n_periods = 5; // Expect: Jan 2, 3, 4, 5, 6
        let dates_list = BDatesList::from_n_periods(start_str.clone(), freq, n_periods)?;

        // check start_date_str (should be original)
        assert_eq!(dates_list.start_date_str(), start_str);
        // check end_date_str (should be the last generated date)
        assert_eq!(dates_list.end_date_str(), "2023-01-06");
        // check frequency enum
        assert_eq!(dates_list.freq(), freq);
        // check frequency string
        assert_eq!(dates_list.freq_str(), "D");

        // Check parsed dates
        assert_eq!(dates_list.start_date()?, date(2023, 1, 1));
        assert_eq!(dates_list.end_date()?, date(2023, 1, 6));

        // Check the actual list matches
        assert_eq!(
            dates_list.list()?,
            vec![
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 4),
                date(2023, 1, 5),
                date(2023, 1, 6)
            ]
        );
        assert_eq!(dates_list.count()?, 5);

        Ok(())
    }

    #[test]
    fn test_bdates_list_from_n_periods_zero_periods() {
        let start_str = "2023-01-01".to_string();
        let freq = BDateFreq::Daily;
        let n_periods = 0;
        let result = BDatesList::from_n_periods(start_str.clone(), freq, n_periods);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "n_periods must be greater than 0"
        );
    }

    #[test]
    fn test_bdates_list_from_n_periods_invalid_start_date() {
        let start_str = "invalid-date".to_string();
        let freq = BDateFreq::Daily;
        let n_periods = 5;
        let result = BDatesList::from_n_periods(start_str.clone(), freq, n_periods);
        assert!(result.is_err());
        // Error comes from NaiveDate::parse_from_str
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("input contains invalid characters")
        );
    }

    #[test]
    fn test_bdates_list_invalid_date_string_new() {
        let dates_list_start_invalid = BDatesList::new(
            "invalid-date".to_string(),
            "2023-12-31".to_string(),
            BDateFreq::Daily,
        );
        assert!(dates_list_start_invalid.list().is_err());
        assert!(dates_list_start_invalid.count().is_err());
        assert!(dates_list_start_invalid.groups().is_err());
        assert!(dates_list_start_invalid.start_date().is_err());
        assert!(dates_list_start_invalid.end_date().is_ok()); // End date is valid

        let dates_list_end_invalid = BDatesList::new(
            "2023-01-01".to_string(),
            "invalid-date".to_string(),
            BDateFreq::Daily,
        );
        assert!(dates_list_end_invalid.list().is_err());
        assert!(dates_list_end_invalid.count().is_err());
        assert!(dates_list_end_invalid.groups().is_err());
        assert!(dates_list_end_invalid.start_date().is_ok()); // Start date is valid
        assert!(dates_list_end_invalid.end_date().is_err());
    }

    // --- BDatesList Core Logic Tests (via list and count) ---

    #[test]
    /// Tests the `list()` method for QuarterEnd frequency over a full year.
    fn test_bdates_list_quarterly_end_list() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-01-01".to_string(),
            "2023-12-31".to_string(),
            BDateFreq::QuarterEnd,
        );

        let list = dates_list.list()?;
        assert_eq!(list.len(), 4);
        assert_eq!(
            list,
            vec![
                date(2023, 3, 31),
                date(2023, 6, 30),
                date(2023, 9, 29),
                date(2023, 12, 29)
            ]
        ); // Fri, Fri, Fri, Fri

        Ok(())
    }

    #[test]
    /// Tests the `list()` method for WeeklyMonday frequency.
    fn test_bdates_list_weekly_monday_list() -> Result<(), Box<dyn Error>> {
        // Range includes start date that is Monday, end date that is Sunday
        let dates_list = BDatesList::new(
            "2023-10-30".to_string(), // Monday (Week 44)
            "2023-11-12".to_string(), // Sunday (Week 45 ends, Week 46 starts)
            BDateFreq::WeeklyMonday,
        );

        let list = dates_list.list()?;
        // Mondays >= 2023-10-30 and <= 2023-11-12:
        // 2023-10-30 (Included)
        // 2023-11-06 (Included)
        // 2023-11-13 (Excluded)
        assert_eq!(list.len(), 2);
        assert_eq!(list, vec![date(2023, 10, 30), date(2023, 11, 6)]);

        Ok(())
    }

    #[test]
    /// Tests the `list()` method for Daily frequency over a short range including weekends.
    fn test_bdates_list_daily_list() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-11-01".to_string(), // Wednesday
            "2023-11-05".to_string(), // Sunday
            BDateFreq::Daily,
        );

        let list = dates_list.list()?;
        // Business days in range: Wed, Thu, Fri
        assert_eq!(list.len(), 3);
        assert_eq!(
            list,
            vec![date(2023, 11, 1), date(2023, 11, 2), date(2023, 11, 3)]
        );

        Ok(())
    }

    #[test]
    /// Tests the `list()` method with an empty date range (end before start).
    fn test_bdates_list_empty_range_list() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-12-31".to_string(),
            "2023-01-01".to_string(), // End date before start date
            BDateFreq::Daily,
        );
        let list = dates_list.list()?;
        assert!(list.is_empty());
        assert_eq!(dates_list.count()?, 0); // Also test count here

        Ok(())
    }

    #[test]
    /// Tests the `count()` method for various frequencies.
    fn test_bdates_list_count() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-01-01".to_string(),
            "2023-12-31".to_string(),
            BDateFreq::MonthEnd,
        );
        assert_eq!(dates_list.count()?, 12); // 12 month ends in 2023

        let dates_list_weekly = BDatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-30".to_string(), // Thu
            BDateFreq::WeeklyFriday,
        );
        // Fridays in range: 2023-11-03, 2023-11-10, 2023-11-17, 2023-11-24
        assert_eq!(dates_list_weekly.count()?, 4);

        Ok(())
    }

    #[test]
    /// Tests `list()` and `count()` for YearlyStart frequency.
    fn test_bdates_list_yearly_start() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-06-01".to_string(),
            "2025-06-01".to_string(),
            BDateFreq::YearStart,
        );
        // Year starts >= 2023-06-01 and <= 2025-06-01:
        // 2023-01-02 (Mon, Jan 1st is Sun) -> Excluded (< 2023-06-01)
        // 2024-01-01 (Mon) -> Included
        // 2025-01-01 (Wed) -> Included
        assert_eq!(dates_list.list()?, vec![date(2024, 1, 1), date(2025, 1, 1)]);
        assert_eq!(dates_list.count()?, 2);

        Ok(())
    }

    #[test]
    /// Tests `list()` and `count()` for MonthlyStart frequency.
    fn test_bdates_list_monthly_start() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-11-15".to_string(), // Mid-Nov
            "2024-02-15".to_string(), // Mid-Feb
            BDateFreq::MonthStart,
        );
        // Month starts >= 2023-11-15 and <= 2024-02-15:
        // 2023-11-01 (Wed) -> Excluded (< 2023-11-15)
        // 2023-12-01 (Fri) -> Included
        // 2024-01-01 (Mon) -> Included
        // 2024-02-01 (Thu) -> Included
        // 2024-03-01 (Fri) -> Excluded (> 2024-02-15)
        assert_eq!(
            dates_list.list()?,
            vec![date(2023, 12, 1), date(2024, 1, 1), date(2024, 2, 1)]
        );
        assert_eq!(dates_list.count()?, 3);

        Ok(())
    }

    #[test]
    /// Tests `list()` and `count()` for WeeklyFriday with a range ending mid-week.
    fn test_bdates_list_weekly_friday_midweek_end() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-11-01".to_string(), // Wed (Week 44)
            "2023-11-14".to_string(), // Tue (Week 46 starts on Mon 13th)
            BDateFreq::WeeklyFriday,
        );
        // Fridays >= 2023-11-01 and <= 2023-11-14:
        // 2023-11-03 (Week 44) -> Included
        // 2023-11-10 (Week 45) -> Included
        // 2023-11-17 (Week 46) -> Excluded (> 2023-11-14)
        assert_eq!(
            dates_list.list()?,
            vec![date(2023, 11, 3), date(2023, 11, 10)]
        );
        assert_eq!(dates_list.count()?, 2);

        Ok(())
    }

    // --- Tests for groups() method ---

    #[test]
    /// Tests the `groups()` method for MonthlyEnd frequency across year boundary.
    fn test_bdates_list_groups_monthly_end() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-10-15".to_string(), // Mid-October
            "2024-01-15".to_string(), // Mid-January next year
            BDateFreq::MonthEnd,
        );

        let groups = dates_list.groups()?;
        // Expected Month Ends within range ["2023-10-15", "2024-01-15"]:
        // 2023-10-31 (>= 2023-10-15) -> Included
        // 2023-11-30 (>= 2023-10-15) -> Included
        // 2023-12-29 (>= 2023-10-15) -> Included
        // 2024-01-31 (> 2024-01-15) -> Excluded
        assert_eq!(groups.len(), 3);

        // Check groups and dates within them (should be sorted by key, then by date).
        // Keys: Monthly(2023, 10), Monthly(2023, 11), Monthly(2023, 12)
        assert_eq!(groups[0], vec![date(2023, 10, 31)]); // Oct 2023 end
        assert_eq!(groups[1], vec![date(2023, 11, 30)]); // Nov 2023 end
        assert_eq!(groups[2], vec![date(2023, 12, 29)]); // Dec 2023 end (31st is Sunday)

        Ok(())
    }

    #[test]
    /// Tests the `groups()` method for Daily frequency over a short range.
    fn test_bdates_list_groups_daily() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-05".to_string(), // Sun
            BDateFreq::Daily,
        );

        let groups = dates_list.groups()?;
        // Business days in range: Wed, Thu, Fri. Each is its own group.
        assert_eq!(groups.len(), 3);

        // Keys: Daily(2023-11-01), Daily(2023-11-02), Daily(2023-11-03)
        assert_eq!(groups[0], vec![date(2023, 11, 1)]);
        assert_eq!(groups[1], vec![date(2023, 11, 2)]);
        assert_eq!(groups[2], vec![date(2023, 11, 3)]);

        Ok(())
    }

    #[test]
    /// Tests the `groups()` method for WeeklyFriday frequency.
    fn test_bdates_list_groups_weekly_friday() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-11-01".to_string(), // Wed (ISO Week 44)
            "2023-11-15".to_string(), // Wed (ISO Week 46)
            BDateFreq::WeeklyFriday,
        );

        let groups = dates_list.groups()?;
        // Fridays in range ["2023-11-01", "2023-11-15"]:
        // 2023-11-03 (ISO Week 44) -> Included
        // 2023-11-10 (ISO Week 45) -> Included
        // 2023-11-17 (ISO Week 46) -> Excluded (> 2023-11-15)
        assert_eq!(groups.len(), 2); // Groups for Week 44, Week 45

        // Check grouping by ISO week
        // Keys: Weekly(2023, 44), Weekly(2023, 45)
        assert_eq!(groups[0], vec![date(2023, 11, 3)]); // ISO Week 44 group
        assert_eq!(groups[1], vec![date(2023, 11, 10)]); // ISO Week 45 group

        Ok(())
    }

    #[test]
    /// Tests the `groups()` method for QuarterlyStart frequency spanning years.
    fn test_bdates_list_groups_quarterly_start_spanning_years() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-08-01".to_string(), // Start date after Q3 2023 start business day
            "2024-05-01".to_string(), // End date after Q2 2024 start business day
            BDateFreq::QuarterStart,
        );

        let groups = dates_list.groups()?;
        // Quarterly starting business days *within the date range* ["2023-08-01", "2024-05-01"]:
        // 2023-07-03 (Q3 2023 start) -> Excluded by start_date 2023-08-01
        // 2023-10-02 (Q4 2023 start - Oct 1st is Sunday) -> Included
        // 2024-01-01 (Q1 2024 start - Jan 1st is Monday) -> Included
        // 2024-04-01 (Q2 2024 start) -> Included
        // 2024-07-01 (Q3 2024 start) -> Excluded by end_date 2024-05-01

        // Expected groups: Q4 2023, Q1 2024, Q2 2024
        assert_eq!(groups.len(), 3);

        // Check groups and dates within them (should be sorted by key, then by date)
        // Key order: Quarterly(2023, 4), Quarterly(2024, 1), Quarterly(2024, 2)
        assert_eq!(groups[0], vec![date(2023, 10, 2)]); // Q4 2023 group
        assert_eq!(groups[1], vec![date(2024, 1, 1)]); // Q1 2024 group (Jan 1st 2024 was a Mon)
        assert_eq!(groups[2], vec![date(2024, 4, 1)]); // Q2 2024 group

        Ok(())
    }

    #[test]
    /// Tests the `groups()` method for YearlyEnd frequency across year boundary.
    fn test_bdates_list_groups_yearly_end() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2022-01-01".to_string(),
            "2024-03-31".to_string(), // End date is Q1 2024
            BDateFreq::YearEnd,
        );

        let groups = dates_list.groups()?;
        // Yearly ending business days *within the date range* ["2022-01-01", "2024-03-31"]:
        // 2022-12-30 (Year 2022 end - 31st Sat) -> Included (>= 2022-01-01)
        // 2023-12-29 (Year 2023 end - 31st Sun) -> Included (>= 2022-01-01)
        // 2024-12-31 (Year 2024 end) -> Excluded because it's after 2024-03-31

        // Expected groups: 2022, 2023
        assert_eq!(groups.len(), 2);

        // Check groups and dates within them (should be sorted by key, then by date)
        // Key order: Yearly(2022), Yearly(2023)
        assert_eq!(groups[0], vec![date(2022, 12, 30)]); // 2022 YE group
        assert_eq!(groups[1], vec![date(2023, 12, 29)]); // 2023 YE group

        Ok(())
    }

    #[test]
    /// Tests the `groups()` method with an empty date range (end before start).
    fn test_bdates_list_groups_empty_range() -> Result<(), Box<dyn Error>> {
        let dates_list = BDatesList::new(
            "2023-12-31".to_string(),
            "2023-01-01".to_string(), // End date before start date
            BDateFreq::Daily,
        );
        let groups = dates_list.groups()?;
        assert!(groups.is_empty());

        Ok(())
    }

    // --- Tests for internal helper functions ---

    #[test]
    /// Tests the `is_weekday` function for all days of the week.
    fn test_is_weekday() {
        assert!(is_weekday(date(2023, 11, 6))); // Mon
        assert!(is_weekday(date(2023, 11, 7))); // Tue
        assert!(is_weekday(date(2023, 11, 8))); // Wed
        assert!(is_weekday(date(2023, 11, 9))); // Thu
        assert!(is_weekday(date(2023, 11, 10))); // Fri
        assert!(!is_weekday(date(2023, 11, 11))); // Sat
        assert!(!is_weekday(date(2023, 11, 12))); // Sun
    }

    #[test]
    /// Tests the `move_to_weekday_on_or_after` function.
    fn test_move_to_weekday_on_or_after() {
        // Already the target weekday
        assert_eq!(
            move_to_weekday_on_or_after(date(2023, 11, 6), Weekday::Mon),
            date(2023, 11, 6)
        );
        // Target weekday is later in the week
        assert_eq!(
            move_to_weekday_on_or_after(date(2023, 11, 8), Weekday::Fri),
            date(2023, 11, 10)
        );
        // Target weekday is next week
        assert_eq!(
            move_to_weekday_on_or_after(date(2023, 11, 11), Weekday::Mon),
            date(2023, 11, 13)
        ); // Sat to next Mon
        assert_eq!(
            move_to_weekday_on_or_after(date(2023, 11, 10), Weekday::Mon),
            date(2023, 11, 13)
        ); // Fri to next Mon
    }

    #[test]
    /// Tests `first_business_day_of_month` including weekend starts.
    fn test_first_business_day_of_month() {
        // Month starts on a weekday
        assert_eq!(first_business_day_of_month(2023, 11), date(2023, 11, 1)); // Nov 1st 2023 is Wed
        // Month starts on a Sunday, 1st business day is Monday
        assert_eq!(first_business_day_of_month(2023, 10), date(2023, 10, 2)); // Oct 1st 2023 is Sun
        // Month starts on a Saturday, 1st business day is Monday
        assert_eq!(first_business_day_of_month(2022, 10), date(2022, 10, 3)); // Oct 1st 2022 is Sat
    }

    #[test]
    /// Tests `last_business_day_of_month` including weekend ends.
    fn test_last_business_day_of_month() {
        // Month ends on a weekday
        assert_eq!(last_business_day_of_month(2023, 11), date(2023, 11, 30)); // Nov 30th 2023 is Thu
        // Month ends on a Sunday, last business day is Friday
        assert_eq!(last_business_day_of_month(2023, 12), date(2023, 12, 29)); // Dec 31st 2023 is Sun
        // Month ends on a Saturday, last business day is Friday
        assert_eq!(last_business_day_of_month(2022, 12), date(2022, 12, 30)); // Dec 31st 2022 is Sat
        // Month ends on Friday
        assert_eq!(last_business_day_of_month(2023, 3), date(2023, 3, 31)); // Mar 31st 2023 is Fri
    }

    #[test]
    /// Tests `days_in_month` including leap years and different month lengths.
    fn test_days_in_month() {
        assert_eq!(days_in_month(2023, 1), 31); // Jan (31)
        assert_eq!(days_in_month(2023, 2), 28); // Feb (28, non-leap)
        assert_eq!(days_in_month(2024, 2), 29); // Feb (29, leap)
        assert_eq!(days_in_month(2023, 4), 30); // Apr (30)
        assert_eq!(days_in_month(2023, 12), 31); // Dec (31)
    }

    #[test]
    /// Tests the `month_to_quarter` mapping.
    fn test_month_to_quarter() {
        assert_eq!(month_to_quarter(1), 1);
        assert_eq!(month_to_quarter(2), 1);
        assert_eq!(month_to_quarter(3), 1);
        assert_eq!(month_to_quarter(4), 2);
        assert_eq!(month_to_quarter(5), 2);
        assert_eq!(month_to_quarter(6), 2);
        assert_eq!(month_to_quarter(7), 3);
        assert_eq!(month_to_quarter(8), 3);
        assert_eq!(month_to_quarter(9), 3);
        assert_eq!(month_to_quarter(10), 4);
        assert_eq!(month_to_quarter(11), 4);
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
    /// Tests `first_business_day_of_quarter` including weekend starts.
    fn test_first_business_day_of_quarter() {
        // Q1 2023: Jan 1st 2023 is Sun, 1st bday is Mon Jan 2nd
        assert_eq!(first_business_day_of_quarter(2023, 1), date(2023, 1, 2));
        // Q2 2023: Apr 1st 2023 is Sat, 1st bday is Mon Apr 3rd
        assert_eq!(first_business_day_of_quarter(2023, 2), date(2023, 4, 3));
        // Q3 2023: Jul 1st 2023 is Sat, 1st bday is Mon Jul 3rd
        assert_eq!(first_business_day_of_quarter(2023, 3), date(2023, 7, 3));
        // Q4 2023: Oct 1st 2023 is Sun, 1st bday is Mon Oct 2nd
        assert_eq!(first_business_day_of_quarter(2023, 4), date(2023, 10, 2));
        // Q1 2024: Jan 1st 2024 is Mon, 1st bday is Mon Jan 1st
        assert_eq!(first_business_day_of_quarter(2024, 1), date(2024, 1, 1));
    }

    #[test]
    /// Tests `last_business_day_of_quarter` including weekend ends.
    fn test_last_business_day_of_quarter() {
        // Q1 2023: Ends Mar 31st (Fri), last bday is Mar 31st
        assert_eq!(last_business_day_of_quarter(2023, 1), date(2023, 3, 31));
        // Q2 2023: Ends Jun 30th (Fri), last bday is Jun 30th
        assert_eq!(last_business_day_of_quarter(2023, 2), date(2023, 6, 30));
        // Q3 2023: Ends Sep 30th (Sat), last bday is Sep 29th (Fri)
        assert_eq!(last_business_day_of_quarter(2023, 3), date(2023, 9, 29));
        // Q4 2023: Ends Dec 31st (Sun), last bday is Dec 29th (Fri)
        assert_eq!(last_business_day_of_quarter(2023, 4), date(2023, 12, 29));
    }

    #[test]
    /// Tests `first_business_day_of_year` including weekend starts.
    fn test_first_business_day_of_year() {
        // 2023: Jan 1st is Sun, 1st bday is Jan 2nd (Mon)
        assert_eq!(first_business_day_of_year(2023), date(2023, 1, 2));
        // 2024: Jan 1st is Mon, 1st bday is Jan 1st (Mon)
        assert_eq!(first_business_day_of_year(2024), date(2024, 1, 1));
        // 2022: Jan 1st is Sat, 1st bday is Jan 3rd (Mon)
        assert_eq!(first_business_day_of_year(2022), date(2022, 1, 3));
    }

    #[test]
    /// Tests `last_business_day_of_year` including weekend ends.
    fn test_last_business_day_of_year() {
        // 2023: Dec 31st is Sun, last bday is Dec 29th (Fri)
        assert_eq!(last_business_day_of_year(2023), date(2023, 12, 29));
        // 2024: Dec 31st is Tue, last bday is Dec 31st (Tue)
        assert_eq!(last_business_day_of_year(2024), date(2024, 12, 31));
        // 2022: Dec 31st is Sat, last bday is Dec 30th (Fri)
        assert_eq!(last_business_day_of_year(2022), date(2022, 12, 30));
    }

    // Test `collect_daily` edge cases
    #[test]
    fn test_collect_daily_single_day_range() {
        // Single weekday
        let start = date(2023, 11, 8); // Wed
        assert_eq!(collect_daily(start, start), vec![start]);
        // Single weekend day - should be empty
        let start = date(2023, 11, 11); // Sat
        assert_eq!(collect_daily(start, start), vec![]);
    }

    #[test]
    fn test_collect_daily_range_spanning_weekend() {
        let start = date(2023, 11, 10); // Fri
        let end = date(2023, 11, 13); // Mon
        // Fri, Sat(skipped), Sun(skipped), Mon
        assert_eq!(
            collect_daily(start, end),
            vec![date(2023, 11, 10), date(2023, 11, 13)]
        );
    }

    // Test `collect_weekly` edge cases
    #[test]
    fn test_collect_weekly_start_is_target() {
        let start = date(2023, 11, 13); // Mon
        let end = date(2023, 11, 20); // Mon
        // Start date is already the target weekday
        assert_eq!(
            collect_weekly(start, end, Weekday::Mon),
            vec![date(2023, 11, 13), date(2023, 11, 20)]
        );
    }

    #[test]
    fn test_collect_weekly_end_before_target() {
        let start = date(2023, 11, 13); // Mon
        let end = date(2023, 11, 16); // Thu
        // Target Friday is after the end date
        assert_eq!(collect_weekly(start, end, Weekday::Fri), vec![]);
    }

    #[test]
    fn test_collect_weekly_single_week() {
        let start = date(2023, 11, 8); // Wed
        let end = date(2023, 11, 14); // Tue
        // Only one Monday (Nov 13) and one Friday (Nov 10) in this range
        assert_eq!(
            collect_weekly(start, end, Weekday::Mon),
            vec![date(2023, 11, 13)]
        );
        assert_eq!(
            collect_weekly(start, end, Weekday::Fri),
            vec![date(2023, 11, 10)]
        );
    }

    // Test `collect_monthly` edge cases
    #[test]
    fn test_collect_monthly_range_starts_mid_month_ends_mid_month() {
        let start = date(2023, 10, 15); // Mid Oct
        let end = date(2024, 1, 15); // Mid Jan
        // Month starts >= start_date AND <= end_date: Nov 2023, Dec 2023, Jan 2024
        assert_eq!(
            collect_monthly(start, end, true),
            vec![date(2023, 11, 1), date(2023, 12, 1), date(2024, 1, 1)]
        );
        // Month ends >= start_date AND <= end_date: Oct 2023, Nov 2023, Dec 2023
        // Last business day of Oct 2023 is Oct 31st, which is after Oct 15th start.
        // Last business day of Jan 2024 is Jan 31st, which is after Jan 15th end.
        assert_eq!(
            collect_monthly(start, end, false),
            vec![date(2023, 10, 31), date(2023, 11, 30), date(2023, 12, 29)]
        );
    }

    #[test]
    fn test_collect_monthly_single_month() {
        let start = date(2023, 11, 1); // Nov 1st (Wed)
        let end = date(2023, 11, 30); // Nov 30th (Thu)
        // Range covers exactly one month, start and end dates are the start/end business days
        assert_eq!(collect_monthly(start, end, true), vec![date(2023, 11, 1)]);
        assert_eq!(collect_monthly(start, end, false), vec![date(2023, 11, 30)]);
    }

    #[test]
    fn test_collect_monthly_range_short() {
        let start = date(2023, 11, 15); // Mid Nov
        let end = date(2023, 11, 20); // Mid Nov
        // No month starts or ends are within this short range.
        assert_eq!(collect_monthly(start, end, true), vec![]);
        assert_eq!(collect_monthly(start, end, false), vec![]);
    }

    #[test]
    fn test_collect_monthly_full_year_start() {
        let start = date(2023, 1, 1);
        let end = date(2023, 12, 31);
        let expected: Vec<NaiveDate> = (1..=12)
            .map(|m| first_business_day_of_month(2023, m))
            .collect();
        assert_eq!(collect_monthly(start, end, true), expected);
    }

    #[test]
    fn test_collect_monthly_full_year_end() {
        let start = date(2023, 1, 1);
        let end = date(2023, 12, 31);
        let expected: Vec<NaiveDate> = (1..=12)
            .map(|m| last_business_day_of_month(2023, m))
            .collect();
        assert_eq!(collect_monthly(start, end, false), expected);
    }

    // Test `collect_quarterly` edge cases
    #[test]
    fn test_collect_quarterly_range_starts_mid_quarter_ends_mid_quarter() {
        let start = date(2023, 8, 15); // Mid Q3 2023
        let end = date(2024, 2, 15); // Mid Q1 2024
        // Q starts >= start_date AND <= end_date: Q4 2023, Q1 2024
        // Q3 2023 start bday (Jul 3rd) < start_date (Aug 15th) -> Excluded
        // Q4 2023 start bday (Oct 2nd) >= start_date (Aug 15th) -> Included
        // Q1 2024 start bday (Jan 1st) >= start_date (Aug 15th) AND <= end_date -> Included
        // Q2 2024 start bday (Apr 1st) > end_date (Feb 15th) -> Excluded
        assert_eq!(
            collect_quarterly(start, end, true),
            vec![date(2023, 10, 2), date(2024, 1, 1)]
        );
        // Q ends >= start_date AND <= end_date: Q3 2023, Q4 2023
        // Q3 2023 end bday (Sep 29th) >= start_date (Aug 15th) -> Included
        // Q4 2023 end bday (Dec 29th) >= start_date (Aug 15th) -> Included
        // Q1 2024 end bday (Mar 29th) > end_date (Feb 15th) -> Excluded
        assert_eq!(
            collect_quarterly(start, end, false),
            vec![date(2023, 9, 29), date(2023, 12, 29)]
        );
    }

    #[test]
    fn test_collect_quarterly_single_quarter() {
        let start = date(2023, 4, 3); // Apr 3rd (Q2 start bday)
        let end = date(2023, 6, 30); // Jun 30th (Q2 end bday)
        // Range covers exactly one quarter
        assert_eq!(collect_quarterly(start, end, true), vec![date(2023, 4, 3)]);
        assert_eq!(
            collect_quarterly(start, end, false),
            vec![date(2023, 6, 30)]
        );
    }

    #[test]
    fn test_collect_quarterly_range_short() {
        let start = date(2023, 5, 15); // Mid Q2
        let end = date(2023, 6, 15); // Mid Q2
        // No quarter starts or ends are within this short range.
        assert_eq!(collect_quarterly(start, end, true), vec![]);
        assert_eq!(collect_quarterly(start, end, false), vec![]);
    }

    #[test]
    fn test_collect_quarterly_full_year_start() {
        let start = date(2023, 1, 1);
        let end = date(2023, 12, 31);
        // Q1: Jan 2, Q2: Apr 3, Q3: Jul 3, Q4: Oct 2
        assert_eq!(
            collect_quarterly(start, end, true),
            vec![
                date(2023, 1, 2),
                date(2023, 4, 3),
                date(2023, 7, 3),
                date(2023, 10, 2)
            ]
        );
    }

    #[test]
    fn test_collect_quarterly_full_year_end() {
        let start = date(2023, 1, 1);
        let end = date(2023, 12, 31);
        // Q1: Mar 31, Q2: Jun 30, Q3: Sep 29, Q4: Dec 29
        assert_eq!(
            collect_quarterly(start, end, false),
            vec![
                date(2023, 3, 31),
                date(2023, 6, 30),
                date(2023, 9, 29),
                date(2023, 12, 29)
            ]
        );
    }

    // Test `collect_yearly` edge cases
    #[test]
    fn test_collect_yearly_range_starts_mid_year_ends_mid_year() -> Result<(), Box<dyn Error>> {
        let start = date(2023, 6, 1); // Mid 2023
        let end = date(2024, 6, 1); // Mid 2024
        // Year starts >= start_date AND <= end_date: 2024
        // 2023 start bday (Jan 2nd) < start_date (Jun 1st) -> Excluded
        // 2024 start bday (Jan 1st) >= start_date (Jun 1st) AND <= end_date -> Included
        // 2025 start bday (Jan 1st) > end_date (Jun 1st) -> Excluded
        assert_eq!(collect_yearly(start, end, true), vec![date(2024, 1, 1)]);
        // Year ends >= start_date AND <= end_date: 2023
        // 2023 end bday (Dec 29th) >= start_date (Jun 1st) -> Included
        // 2024 end bday (Dec 31st) > end_date (Jun 1st) -> Excluded <-- Correction: Original thought was wrong
        assert_eq!(collect_yearly(start, end, false), vec![date(2023, 12, 29)]);
        Ok(())
    }

    #[test]
    fn test_collect_yearly_single_year() {
        let start = date(2024, 1, 1); // 2024 start bday
        let end = date(2024, 12, 31); // 2024 end bday
        // Range covers exactly one year
        assert_eq!(collect_yearly(start, end, true), vec![date(2024, 1, 1)]);
        assert_eq!(collect_yearly(start, end, false), vec![date(2024, 12, 31)]);
    }

    #[test]
    fn test_collect_yearly_range_short() {
        let start = date(2023, 5, 15); // Mid 2023
        let end = date(2023, 6, 15); // Mid 2023
        // No year starts or ends are within this short range.
        assert_eq!(collect_yearly(start, end, true), vec![]);
        assert_eq!(collect_yearly(start, end, false), vec![]);
    }

    #[test]
    fn test_collect_yearly_full_years() {
        let start = date(2022, 1, 1);
        let end = date(2024, 12, 31);
        // Year starts
        assert_eq!(
            collect_yearly(start, end, true),
            vec![date(2022, 1, 3), date(2023, 1, 2), date(2024, 1, 1)]
        );
        // Year ends
        assert_eq!(
            collect_yearly(start, end, false),
            vec![date(2022, 12, 30), date(2023, 12, 29), date(2024, 12, 31)]
        );
    }

    // --- Tests for Generator Helper Functions ---

    #[test]
    fn test_find_first_bdate_on_or_after() {
        // Daily
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 8), BDateFreq::Daily),
            date(2023, 11, 8)
        ); // Wed -> Wed
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 11), BDateFreq::Daily),
            date(2023, 11, 13)
        ); // Sat -> Mon

        // Weekly Mon
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 8), BDateFreq::WeeklyMonday),
            date(2023, 11, 13)
        ); // Wed -> Next Mon
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 13), BDateFreq::WeeklyMonday),
            date(2023, 11, 13)
        ); // Mon -> Mon
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 12), BDateFreq::WeeklyMonday),
            date(2023, 11, 13)
        ); // Sun -> Mon

        // Weekly Fri
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 8), BDateFreq::WeeklyFriday),
            date(2023, 11, 10)
        ); // Wed -> Fri
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 10), BDateFreq::WeeklyFriday),
            date(2023, 11, 10)
        ); // Fri -> Fri
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 11), BDateFreq::WeeklyFriday),
            date(2023, 11, 17)
        ); // Sat -> Next Fri

        // Month Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 1), BDateFreq::MonthStart),
            date(2023, 11, 1)
        ); // Nov 1 (Wed) -> Nov 1
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 10, 15), BDateFreq::MonthStart),
            date(2023, 11, 1)
        ); // Mid Oct -> Nov 1
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 15), BDateFreq::MonthStart),
            date(2024, 1, 1)
        ); // Mid Dec -> Jan 1
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 10, 1), BDateFreq::MonthStart),
            date(2023, 10, 2)
        ); // Oct 1 (Sun) -> Oct 2

        // Month End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 30), BDateFreq::MonthEnd),
            date(2023, 11, 30)
        ); // Nov 30 (Thu) -> Nov 30
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 15), BDateFreq::MonthEnd),
            date(2023, 11, 30)
        ); // Mid Nov -> Nov 30
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 30), BDateFreq::MonthEnd),
            date(2024, 1, 31)
        ); // Dec 30 (Sat) -> Jan 31 (Dec end was 29th, which is < 30th)
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 29), BDateFreq::MonthEnd),
            date(2023, 12, 29)
        ); // Dec 29 (Fri) -> Dec 29
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 9, 30), BDateFreq::MonthEnd),
            date(2023, 10, 31)
        ); // Sep 30 (Sat) -> Oct 31 (Sep end was 29th, < 30th)

        // Quarter Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 10, 2), BDateFreq::QuarterStart),
            date(2023, 10, 2)
        ); // Q4 Start (Mon) -> Q4 Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 8, 15), BDateFreq::QuarterStart),
            date(2023, 10, 2)
        ); // Mid Q3 -> Q4 Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 11, 15), BDateFreq::QuarterStart),
            date(2024, 1, 1)
        ); // Mid Q4 -> Q1 Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 1, 1), BDateFreq::QuarterStart),
            date(2023, 1, 2)
        ); // Jan 1 (Sun) -> Jan 2 (Mon)

        // Quarter End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 9, 29), BDateFreq::QuarterEnd),
            date(2023, 9, 29)
        ); // Q3 End (Fri) -> Q3 End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 8, 15), BDateFreq::QuarterEnd),
            date(2023, 9, 29)
        ); // Mid Q3 -> Q3 End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 10, 15), BDateFreq::QuarterEnd),
            date(2023, 12, 29)
        ); // Mid Q4 -> Q4 End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 30), BDateFreq::QuarterEnd),
            date(2024, 3, 29)
        ); // Dec 30 (Sat) -> Q1 End (Q4 end was 29th, < 30th)

        // Year Start
        assert_eq!(
            find_first_bdate_on_or_after(date(2024, 1, 1), BDateFreq::YearStart),
            date(2024, 1, 1)
        ); // Jan 1 (Mon) -> Jan 1
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 6, 15), BDateFreq::YearStart),
            date(2024, 1, 1)
        ); // Mid 2023 -> Jan 1 2024
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 1, 1), BDateFreq::YearStart),
            date(2023, 1, 2)
        ); // Jan 1 (Sun) -> Jan 2

        // Year End
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 29), BDateFreq::YearEnd),
            date(2023, 12, 29)
        ); // Dec 29 (Fri) -> Dec 29
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 6, 15), BDateFreq::YearEnd),
            date(2023, 12, 29)
        ); // Mid 2023 -> Dec 29 2023
        assert_eq!(
            find_first_bdate_on_or_after(date(2023, 12, 30), BDateFreq::YearEnd),
            date(2024, 12, 31)
        ); // Dec 30 (Sat) -> Dec 31 2024 (2023 end was 29th, < 30th)
    }

    #[test]
    fn test_find_next_bdate() {
        // Daily
        assert_eq!(
            find_next_bdate(date(2023, 11, 8), BDateFreq::Daily),
            date(2023, 11, 9)
        ); // Wed -> Thu
        assert_eq!(
            find_next_bdate(date(2023, 11, 10), BDateFreq::Daily),
            date(2023, 11, 13)
        ); // Fri -> Mon

        // Weekly Mon
        assert_eq!(
            find_next_bdate(date(2023, 11, 13), BDateFreq::WeeklyMonday),
            date(2023, 11, 20)
        ); // Mon -> Next Mon

        // Weekly Fri
        assert_eq!(
            find_next_bdate(date(2023, 11, 10), BDateFreq::WeeklyFriday),
            date(2023, 11, 17)
        ); // Fri -> Next Fri

        // Month Start
        assert_eq!(
            find_next_bdate(date(2023, 11, 1), BDateFreq::MonthStart),
            date(2023, 12, 1)
        ); // Nov 1 -> Dec 1
        assert_eq!(
            find_next_bdate(date(2023, 12, 1), BDateFreq::MonthStart),
            date(2024, 1, 1)
        ); // Dec 1 -> Jan 1

        // Month End
        assert_eq!(
            find_next_bdate(date(2023, 10, 31), BDateFreq::MonthEnd),
            date(2023, 11, 30)
        ); // Oct 31 -> Nov 30
        assert_eq!(
            find_next_bdate(date(2023, 11, 30), BDateFreq::MonthEnd),
            date(2023, 12, 29)
        ); // Nov 30 -> Dec 29
        assert_eq!(
            find_next_bdate(date(2023, 12, 29), BDateFreq::MonthEnd),
            date(2024, 1, 31)
        ); // Dec 29 -> Jan 31

        // Quarter Start
        assert_eq!(
            find_next_bdate(date(2023, 10, 2), BDateFreq::QuarterStart),
            date(2024, 1, 1)
        ); // Q4 Start -> Q1 Start
        assert_eq!(
            find_next_bdate(date(2024, 1, 1), BDateFreq::QuarterStart),
            date(2024, 4, 1)
        ); // Q1 Start -> Q2 Start

        // Quarter End
        assert_eq!(
            find_next_bdate(date(2023, 9, 29), BDateFreq::QuarterEnd),
            date(2023, 12, 29)
        ); // Q3 End -> Q4 End
        assert_eq!(
            find_next_bdate(date(2023, 12, 29), BDateFreq::QuarterEnd),
            date(2024, 3, 29)
        ); // Q4 End -> Q1 End (Mar 31 2024 is Sun)

        // Year Start
        assert_eq!(
            find_next_bdate(date(2023, 1, 2), BDateFreq::YearStart),
            date(2024, 1, 1)
        ); // 2023 Start -> 2024 Start
        assert_eq!(
            find_next_bdate(date(2024, 1, 1), BDateFreq::YearStart),
            date(2025, 1, 1)
        ); // 2024 Start -> 2025 Start

        // Year End
        assert_eq!(
            find_next_bdate(date(2022, 12, 30), BDateFreq::YearEnd),
            date(2023, 12, 29)
        ); // 2022 End -> 2023 End
        assert_eq!(
            find_next_bdate(date(2023, 12, 29), BDateFreq::YearEnd),
            date(2024, 12, 31)
        ); // 2023 End -> 2024 End
    }

    // --- Tests for BDatesGenerator ---

    #[test]
    fn test_generator_new_zero_periods() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 1, 1);
        let freq = BDateFreq::Daily;
        let n_periods = 0;
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;
        assert_eq!(generator.next(), None); // Should be immediately exhausted
        Ok(())
    }

    #[test]
    fn test_generator_daily() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 10); // Friday
        let freq = BDateFreq::Daily;
        let n_periods = 4;
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 10))); // Fri
        assert_eq!(generator.next(), Some(date(2023, 11, 13))); // Mon
        assert_eq!(generator.next(), Some(date(2023, 11, 14))); // Tue
        assert_eq!(generator.next(), Some(date(2023, 11, 15))); // Wed
        assert_eq!(generator.next(), None); // Exhausted

        // Test starting on weekend
        let start_date_sat = date(2023, 11, 11); // Saturday
        let mut generator_sat = BDatesGenerator::new(start_date_sat, freq, 2)?;
        assert_eq!(generator_sat.next(), Some(date(2023, 11, 13))); // Mon
        assert_eq!(generator_sat.next(), Some(date(2023, 11, 14))); // Tue
        assert_eq!(generator_sat.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_weekly_monday() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 8); // Wednesday
        let freq = BDateFreq::WeeklyMonday;
        let n_periods = 3;
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 13)));
        assert_eq!(generator.next(), Some(date(2023, 11, 20)));
        assert_eq!(generator.next(), Some(date(2023, 11, 27)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_weekly_friday() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 11); // Saturday
        let freq = BDateFreq::WeeklyFriday;
        let n_periods = 3;
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 17)));
        assert_eq!(generator.next(), Some(date(2023, 11, 24)));
        assert_eq!(generator.next(), Some(date(2023, 12, 1)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_month_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 10, 15); // Mid-Oct
        let freq = BDateFreq::MonthStart;
        let n_periods = 4; // Nov, Dec, Jan, Feb
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 11, 1)));
        assert_eq!(generator.next(), Some(date(2023, 12, 1)));
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2024, 2, 1)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_month_end() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 9, 30); // Sep 30 (Sat)
        let freq = BDateFreq::MonthEnd;
        let n_periods = 4; // Oct, Nov, Dec, Jan
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 10, 31))); // Sep end was 29th < 30th, so start with Oct end
        assert_eq!(generator.next(), Some(date(2023, 11, 30)));
        assert_eq!(generator.next(), Some(date(2023, 12, 29)));
        assert_eq!(generator.next(), Some(date(2024, 1, 31)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_quarter_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 8, 1); // Mid-Q3
        let freq = BDateFreq::QuarterStart;
        let n_periods = 3; // Q4'23, Q1'24, Q2'24
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 10, 2))); // Q3 start was Jul 3, < Aug 1. Next is Q4 start.
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2024, 4, 1)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_quarter_end() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 1); // Mid-Q4
        let freq = BDateFreq::QuarterEnd;
        let n_periods = 3; // Q4'23, Q1'24, Q2'24
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 12, 29))); // Q4 end is Dec 29 >= Nov 1
        assert_eq!(generator.next(), Some(date(2024, 3, 29))); // Q1 end (Mar 31 is Sun)
        assert_eq!(generator.next(), Some(date(2024, 6, 28))); // Q2 end (Jun 30 is Sun)
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_year_start() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 1, 1); // Jan 1 (Sun)
        let freq = BDateFreq::YearStart;
        let n_periods = 3; // 2023, 2024, 2025
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 1, 2))); // 2023 start bday >= Jan 1
        assert_eq!(generator.next(), Some(date(2024, 1, 1)));
        assert_eq!(generator.next(), Some(date(2025, 1, 1)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_year_end() -> Result<(), Box<dyn Error>> {
        let start_date = date(2022, 12, 31); // Dec 31 (Sat)
        let freq = BDateFreq::YearEnd;
        let n_periods = 3; // 2023, 2024, 2025
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;

        assert_eq!(generator.next(), Some(date(2023, 12, 29))); // 2022 end was Dec 30 < Dec 31. Next is 2023 end.
        assert_eq!(generator.next(), Some(date(2024, 12, 31)));
        assert_eq!(generator.next(), Some(date(2025, 12, 31)));
        assert_eq!(generator.next(), None);

        Ok(())
    }

    #[test]
    fn test_generator_collect() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 10); // Friday
        let freq = BDateFreq::Daily;
        let n_periods = 4;
        let generator = BDatesGenerator::new(start_date, freq, n_periods)?; // Use non-mut binding for collect
        let dates: Vec<NaiveDate> = generator.collect();

        assert_eq!(
            dates,
            vec![
                date(2023, 11, 10), // Fri
                date(2023, 11, 13), // Mon
                date(2023, 11, 14), // Tue
                date(2023, 11, 15)  // Wed
            ]
        );
        Ok(())
    }
} // end mod tests
