use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::result::Result;

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
        Self::from_str(&freq)
    }

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
    pub fn from_str(freq: &str) -> Result<Self, Box<dyn Error>> {
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

/// Represents a list of business dates generated between a start and end date
/// at a specified frequency. Provides methods to retrieve the full list,
/// count, or dates grouped by period.
#[derive(Debug, Clone)]
pub struct BDatesList {
    start_date_str: String,
    end_date_str: String,
    freq: BDateFreq,
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

impl BDatesList {
    /// Creates a new `BDatesList` instance.
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
    /// Returns a `chrono::ParseError` if the start date string is not in
    /// "YYYY-MM-DD" format.
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
    /// Returns a `chrono::ParseError` if the end date string is not in
    /// "YYYY-MM-DD" format.
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
    dates.retain(|d| is_weekday(*d));

    // Ensure the final list is sorted. The `collect_*` functions generally
    // produce sorted output, but an explicit sort guarantees it.
    dates.sort();

    Ok(dates)
}

/* ---------------------- Low-Level Date Collection Functions (Internal) ---------------------- */

/// Returns all business days (Mon-Fri) day-by-day within the range.
fn collect_daily(start_date: NaiveDate, end_date: NaiveDate) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut current = start_date;
    while current <= end_date {
        if is_weekday(current) {
            result.push(current);
        }
        // Use succ_opt() and unwrap(), assuming valid date range and no overflow
        current = current.succ_opt().unwrap();
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
        result.push(current);
        // Use checked_add_signed for safety, though basic addition is likely fine for 7 days.
        current = current
            .checked_add_signed(Duration::days(7))
            .expect("date overflow");
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
        // Use _opt and unwrap(), expecting valid month/year combinations within realistic ranges.
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
            result.push(candidate);
        }
        // Note: We don't break if candidate < start_date because a later month's candidate
        // might be within the range.

        // Advance to the next month.
        let (ny, nm) = next_month((year, month));
        year = ny;
        month = nm;

        // Optimization: Stop if we have moved clearly past the end date's year.
        // If the year matches, we need to check the month.
        if year > end_date.year() {
            break;
        }
        if year == end_date.year() && month > end_date.month() {
            break;
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
        // Use _opt and unwrap(), expecting valid quarter/year combinations.
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
            result.push(candidate);
        }
        // Note: We don't break if candidate < start_date because a later quarter
        // might be within the range.

        // Advance to the next quarter.
        if q == 4 {
            year += 1;
            q = 1;
        } else {
            q += 1;
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
        // Use _opt and unwrap(), expecting valid year.
        let candidate = if want_first_day {
            first_business_day_of_year(year)
        } else {
            last_business_day_of_year(year)
        };

        // If the candidate is within the specified range [start_date, end_date], add it.
        if candidate >= start_date && candidate <= end_date {
            result.push(candidate);
        } else if candidate > end_date {
            // Optimization: If the candidate for the current year is already past end_date,
            // then candidates for all subsequent years will also be past end_date.
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
        // Use succ_opt() and unwrap(), assuming valid date and no overflow
        current = current.succ_opt().unwrap();
    }
    current
}

/// Return the earliest business day of the given (year, month).
fn first_business_day_of_month(year: i32, month: u32) -> NaiveDate {
    // Start with the 1st of the month. Use _opt and unwrap(), assuming valid Y/M.
    let mut d = NaiveDate::from_ymd_opt(year, month, 1).expect("invalid year-month combination");
    // If it’s Sat/Sun, move forward until we find a weekday.
    while !is_weekday(d) {
        // Use succ_opt() and unwrap(), assuming valid date and no overflow.
        d = d.succ_opt().unwrap();
    }
    d
}

/// Return the latest business day of the given (year, month).
fn last_business_day_of_month(year: i32, month: u32) -> NaiveDate {
    let last_dom = days_in_month(year, month);
    // Use _opt and unwrap(), assuming valid Y/M/D combination.
    let mut d =
        NaiveDate::from_ymd_opt(year, month, last_dom).expect("invalid year-month-day combination");
    // If it’s Sat/Sun, move backward until we find a weekday.
    while !is_weekday(d) {
        // Use pred_opt() and unwrap(), assuming valid date and no underflow.
        d = d.pred_opt().unwrap();
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
    // Use _opt and unwrap(), assuming valid Y/M combination (start of next month).
    let first_of_next = NaiveDate::from_ymd_opt(ny, nm, 1).expect("invalid next year-month");
    // Use pred_opt() and unwrap(), assuming valid date and no underflow (first of month - 1).
    let last_of_this = first_of_next
        .pred_opt()
        .expect("invalid date before first of month");
    last_of_this.day()
}

/// Converts a month number (1-12) to a quarter number (1-4).
fn month_to_quarter(m: u32) -> u32 {
    (m - 1) / 3 + 1 // Simple integer division for mapping
}

/// Returns the 1st day of the month that starts a given (year, quarter).
fn quarter_to_first_date(year: i32, quarter: u32) -> NaiveDate {
    let month = match quarter {
        1 => 1,
        2 => 4,
        3 => 7,
        4 => 10,
        _ => panic!("invalid quarter: {}", quarter), // This function expects quarter 1-4
    };
    // Use _opt and unwrap(), assuming valid Y/M/D combination (first day of quarter month).
    NaiveDate::from_ymd_opt(year, month, 1).expect("invalid year/month derived from quarter")
}

/// Return the earliest business day in the given (year, quarter).
fn first_business_day_of_quarter(year: i32, quarter: u32) -> NaiveDate {
    let mut d = quarter_to_first_date(year, quarter);
    // If the first day is a weekend, move forward to the next weekday.
    while !is_weekday(d) {
        // Use succ_opt() and unwrap(), assuming valid date and no overflow.
        d = d.succ_opt().unwrap();
    }
    d
}

/// Return the last business day in the given (year, quarter).
fn last_business_day_of_quarter(year: i32, quarter: u32) -> NaiveDate {
    // The last month of a quarter is the start month + 2.
    let start = quarter_to_first_date(year, quarter);
    let last_month_in_quarter = start.month() + 2;
    last_business_day_of_month(year, last_month_in_quarter)
}

/// Returns the earliest business day (Mon-Fri) of the given year.
fn first_business_day_of_year(year: i32) -> NaiveDate {
    // Start with Jan 1st. Use _opt and unwrap(), assuming valid Y/M/D combination.
    let mut d = NaiveDate::from_ymd_opt(year, 1, 1).expect("invalid year for Jan 1st");
    // If Jan 1st is a weekend, move forward to the next weekday.
    while !is_weekday(d) {
        // Use succ_opt() and unwrap(), assuming valid date and no overflow.
        d = d.succ_opt().unwrap();
    }
    d
}

/// Returns the last business day (Mon-Fri) of the given year.
fn last_business_day_of_year(year: i32) -> NaiveDate {
    // Start with Dec 31st. Use _opt and unwrap(), assuming valid Y/M/D combination.
    let mut d = NaiveDate::from_ymd_opt(year, 12, 31).expect("invalid year for Dec 31st");
    // If Dec 31st is a weekend, move backward to the previous weekday.
    while !is_weekday(d) {
        // Use pred_opt() and unwrap(), assuming valid date and no underflow.
        d = d.pred_opt().unwrap();
    }
    d
}

// --- Example Usage and Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    // Helper to create a NaiveDate for tests, handling the unwrap for fixed dates.
    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    // --- BDateFreq Tests ---

    #[test]
    fn test_bdatefreq_from_str() -> Result<(), Box<dyn Error>> {
        assert_eq!(BDateFreq::from_str("D")?, BDateFreq::Daily);
        assert_eq!(BDateFreq::from_str("W")?, BDateFreq::WeeklyMonday);
        assert_eq!(BDateFreq::from_str("M")?, BDateFreq::MonthStart);
        assert_eq!(BDateFreq::from_str("Q")?, BDateFreq::QuarterStart);

        // Test YearStart codes and aliases (Y, A, AS, YS)
        assert_eq!(BDateFreq::from_str("Y")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("A")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("AS")?, BDateFreq::YearStart);
        assert_eq!(BDateFreq::from_str("YS")?, BDateFreq::YearStart);

        assert_eq!(BDateFreq::from_str("ME")?, BDateFreq::MonthEnd);
        assert_eq!(BDateFreq::from_str("QE")?, BDateFreq::QuarterEnd);
        assert_eq!(BDateFreq::from_str("WF")?, BDateFreq::WeeklyFriday);

        // Test YearEnd codes and aliases (YE, AE)
        assert_eq!(BDateFreq::from_str("YE")?, BDateFreq::YearEnd);
        assert_eq!(BDateFreq::from_str("AE")?, BDateFreq::YearEnd);

        // Test aliases for other frequencies
        assert_eq!(BDateFreq::from_str("WS")?, BDateFreq::WeeklyMonday);
        assert_eq!(BDateFreq::from_str("MS")?, BDateFreq::MonthStart);
        assert_eq!(BDateFreq::from_str("QS")?, BDateFreq::QuarterStart);

        // Test invalid string
        assert!(BDateFreq::from_str("INVALID").is_err());
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
    fn test_bdates_list_properties() -> Result<(), Box<dyn Error>> {
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
    fn test_bdates_list_invalid_date_string() {
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
    /// Tests `quarter_to_first_date` for all quarters.
    fn test_quarter_to_first_date() {
        assert_eq!(quarter_to_first_date(2023, 1), date(2023, 1, 1));
        assert_eq!(quarter_to_first_date(2023, 2), date(2023, 4, 1));
        assert_eq!(quarter_to_first_date(2023, 3), date(2023, 7, 1));
        assert_eq!(quarter_to_first_date(2023, 4), date(2023, 10, 1));
        // Panics on invalid quarter
        let result = std::panic::catch_unwind(|| quarter_to_first_date(2023, 5));
        assert!(result.is_err());
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
        // Q1 2024 start bday (Jan 1st) >= start_date (Aug 15th) -> Included
        // Q2 2024 start bday (Apr 1st) > end_date (Feb 15th) -> Excluded
        assert_eq!(
            collect_quarterly(start, end, true),
            vec![date(2023, 10, 2), date(2024, 1, 1)]
        );
        // Q ends >= start_date AND <= end_date: Q3 2023, Q4 2023
        // Q3 2023 end bday (Sep 29th) >= start_date (Aug 15th) -> Included
        // Q4 2023 end bday (Dec 29th) >= start_date (Aug 15th) -> Included
        // Q1 2024 end bday (Mar 31st) > end_date (Feb 15th) -> Excluded
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

    // Test `collect_yearly` edge cases
    #[test]
    fn test_collect_yearly_range_starts_mid_year_ends_mid_year() -> Result<(), Box<dyn Error>> {
        let start = date(2023, 6, 1); // Mid 2023
        let end = date(2024, 6, 1); // Mid 2024
        // Year starts >= start_date AND <= end_date: 2024
        // 2023 start bday (Jan 2nd) < start_date (Jun 1st) -> Excluded
        // 2024 start bday (Jan 1st) >= start_date (Jun 1st) -> Included
        // 2025 start bday (Jan 1st) > end_date (Jun 1st) -> Excluded
        assert_eq!(collect_yearly(start, end, true), vec![date(2024, 1, 1)]);
        // Year ends >= start_date AND <= end_date: 2023
        // 2023 end bday (Dec 29th) >= start_date (Jun 1st) -> Included
        // 2024 end bday (Dec 31st) > end_date (Jun 1st) -> Included
        assert_eq!(
            collect_yearly(start, end, false),
            vec![date(2023, 12, 29)]
        );
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
}
