use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::result::Result;

// --- Enums ---

/// Represents the frequency at which calendar dates should be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateFreq {
    Daily,        // Every single day
    WeeklyMonday, // The Monday of each week
    WeeklyFriday, // The Friday of each week
    MonthStart,   // The 1st day of each month
    MonthEnd,     // The actual last day of each month
    QuarterStart, // The 1st day of each quarter (Jan 1, Apr 1, Jul 1, Oct 1)
    QuarterEnd,   // The actual last day of each quarter (Mar 31, Jun 30, Sep 30, Dec 31)
    YearStart,    // January 1st of each year
    YearEnd,      // December 31st of each year
}

/// Indicates whether the first or last date in a periodic group (like month, quarter)
/// is selected for the frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    Start, // Indicates picking the first date in a group's period.
    End,   // Indicates picking the last date in a group's period.
}

// Helper enum for grouping dates (Internal)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum GroupKey {
    Daily(NaiveDate),    // Group by the specific date (for Daily frequency)
    Weekly(i32, u32),    // Group by year and ISO week number
    Monthly(i32, u32),   // Group by year and month (1-12)
    Quarterly(i32, u32), // Group by year and quarter (1-4)
    Yearly(i32),         // Group by year
}

// --- DateFreq Implementation ---

impl DateFreq {
    /// Attempts to parse a frequency string into a `DateFreq` enum.
    /// Supports various frequency codes and common aliases.
    ///
    /// | Code | Alias   | Description             |
    /// |------|---------|-------------------------|
    /// | D    |         | Daily                   |
    /// | W    | WS      | Weekly Monday           |
    /// | WF   |         | Weekly Friday           |
    /// | M    | MS      | Month Start (1st)       |
    /// | ME   |         | Month End (actual last) |
    /// | Q    | QS      | Quarter Start (1st)     |
    /// | QE   |         | Quarter End (actual last)|
    /// | Y    | A, AS, YS | Year Start (Jan 1st)    |
    /// | YE   | AE      | Year End (Dec 31st)     |
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
            "D" => DateFreq::Daily,
            "W" | "WS" => DateFreq::WeeklyMonday,
            "WF" => DateFreq::WeeklyFriday,
            "M" | "MS" => DateFreq::MonthStart,
            "ME" => DateFreq::MonthEnd,
            "Q" | "QS" => DateFreq::QuarterStart,
            "QE" => DateFreq::QuarterEnd,
            "Y" | "A" | "AS" | "YS" => DateFreq::YearStart,
            "YE" | "AE" => DateFreq::YearEnd,
            _ => return Err(format!("Invalid frequency specified: {}", freq).into()),
        };
        Ok(r)
    }

    /// Attempts to parse a frequency string into a `DateFreq` enum.
    /// Convenience wrapper around `from_str`.
    pub fn from_string(freq: String) -> Result<Self, Box<dyn Error>> {
        Self::from_str(&freq)
    }

    /// Returns the canonical string representation of the frequency.
    pub fn to_string(&self) -> String {
        let r = match self {
            DateFreq::Daily => "D",
            DateFreq::WeeklyMonday => "W",
            DateFreq::WeeklyFriday => "WF",
            DateFreq::MonthStart => "M",
            DateFreq::MonthEnd => "ME",
            DateFreq::QuarterStart => "Q",
            DateFreq::QuarterEnd => "QE",
            DateFreq::YearStart => "Y",
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

// --- DatesList Struct ---

/// Represents a list of calendar dates generated between a start and end date
/// at a specified frequency. Includes all days (weekends, etc.).
/// Provides methods to retrieve the full list, count, or dates grouped by period.
#[derive(Debug, Clone)]
pub struct DatesList {
    start_date_str: String,
    end_date_str: String,
    freq: DateFreq,
}

// --- DatesList Implementation ---

impl DatesList {
    /// Creates a new `DatesList` instance.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The inclusive start date as a string (e.g., "YYYY-MM-DD").
    /// * `end_date_str` - The inclusive end date as a string (e.g., "YYYY-MM-DD").
    /// * `freq` - The frequency (`DateFreq`) for generating dates.
    pub fn new(start_date_str: String, end_date_str: String, freq: DateFreq) -> Self {
        DatesList {
            start_date_str,
            end_date_str,
            freq,
        }
    }

    /// Returns the flat list of calendar dates within the specified range and frequency.
    ///
    /// The list is guaranteed to be sorted chronologically.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn list(&self) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
        get_dates_list_with_freq(&self.start_date_str, &self.end_date_str, self.freq)
    }

    /// Returns the count of calendar dates within the specified range and frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn count(&self) -> Result<usize, Box<dyn Error>> {
        self.list().map(|list| list.len())
    }

    /// Returns a list of date lists, where each inner list contains the dates
    /// generated by `self.list()` that belong to the same period (determined by `self.freq`).
    ///
    /// The outer list (groups) is sorted by period chronologically, and the
    /// inner lists (dates within groups) are also sorted chronologically.
    ///
    /// - For `Daily` frequency, each date forms its own group.
    /// - For `Weekly` frequencies, grouping is by ISO week number/year.
    /// - For `Monthly` frequencies, grouping is by month/year.
    /// - For `Quarterly` frequencies, grouping is by quarter/year.
    /// - For `Yearly` frequencies, grouping is by year.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn groups(&self) -> Result<Vec<Vec<NaiveDate>>, Box<dyn Error>> {
        let dates = self.list()?;
        if dates.is_empty() {
            return Ok(Vec::new());
        }

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
        sorted_groups.sort_by_key(|(k, _)| *k);

        let result_groups = sorted_groups.into_iter().map(|(_, dates)| dates).collect();
        Ok(result_groups)
    }

    /// Returns the start date parsed as a `NaiveDate`.
    pub fn start_date(&self) -> Result<NaiveDate, Box<dyn Error>> {
        NaiveDate::parse_from_str(&self.start_date_str, "%Y-%m-%d").map_err(|e| e.into())
    }

    /// Returns the start date string.
    pub fn start_date_str(&self) -> &str {
        &self.start_date_str
    }

    /// Returns the end date parsed as a `NaiveDate`.
    pub fn end_date(&self) -> Result<NaiveDate, Box<dyn Error>> {
        NaiveDate::parse_from_str(&self.end_date_str, "%Y-%m-%d").map_err(|e| e.into())
    }

    /// Returns the end date string.
    pub fn end_date_str(&self) -> &str {
        &self.end_date_str
    }

    /// Returns the frequency enum (`DateFreq`).
    pub fn freq(&self) -> DateFreq {
        self.freq
    }

    /// Returns the canonical string representation of the frequency.
    pub fn freq_str(&self) -> String {
        self.freq.to_string()
    }
}

// --- Internal Helper Functions ---

/// Generates the flat list of calendar dates for the given range and frequency.
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

    let mut dates = match freq {
        DateFreq::Daily => collect_calendar_daily(start_date, end_date),
        DateFreq::WeeklyMonday => collect_calendar_weekly(start_date, end_date, Weekday::Mon),
        DateFreq::WeeklyFriday => collect_calendar_weekly(start_date, end_date, Weekday::Fri),
        DateFreq::MonthStart => collect_calendar_monthly(start_date, end_date, true),
        DateFreq::MonthEnd => collect_calendar_monthly(start_date, end_date, false),
        DateFreq::QuarterStart => collect_calendar_quarterly(start_date, end_date, true),
        DateFreq::QuarterEnd => collect_calendar_quarterly(start_date, end_date, false),
        DateFreq::YearStart => collect_calendar_yearly(start_date, end_date, true),
        DateFreq::YearEnd => collect_calendar_yearly(start_date, end_date, false),
    };

    // Ensure the final list is sorted (most collectors produce sorted, but good practice).
    dates.sort_unstable(); // Slightly faster for non-pathological cases

    Ok(dates)
}

// --- Internal Date Collection Logic ---

fn collect_calendar_daily(start_date: NaiveDate, end_date: NaiveDate) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut current = start_date;
    while current <= end_date {
        result.push(current);
        match current.succ_opt() {
            Some(next_day) => current = next_day,
            None => break, // Avoid panic on date overflow near max date
        }
    }
    result
}

fn collect_calendar_weekly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    target_weekday: Weekday,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut current = move_to_weekday_on_or_after(start_date, target_weekday);

    while current <= end_date {
        result.push(current);
        match current.checked_add_signed(Duration::days(7)) {
            Some(next_week_day) => current = next_week_day,
            None => break, // Avoid panic on date overflow
        }
    }
    result
}

fn collect_calendar_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut year = start_date.year();
    let mut month = start_date.month();

    let next_month =
        |(yr, mo): (i32, u32)| -> (i32, u32) { if mo == 12 { (yr + 1, 1) } else { (yr, mo + 1) } };

    loop {
        let candidate_res = if want_first_day {
            NaiveDate::from_ymd_opt(year, month, 1)
        } else {
            days_in_month(year, month).and_then(|day| NaiveDate::from_ymd_opt(year, month, day))
        };

        let candidate = match candidate_res {
            Some(date) => date,
            None => {
                // Should not happen with valid year/month logic, but break defensively
                eprintln!("Warning: Invalid date generated for {}-{}", year, month);
                break;
            }
        };

        if candidate > end_date {
            break;
        }
        if candidate >= start_date {
            result.push(candidate);
        }

        let (ny, nm) = next_month((year, month));
        year = ny;
        month = nm;

        if year > end_date.year() {
            break;
        }
        if year == end_date.year() && month > end_date.month() {
            break;
        }
    }
    result
}

fn collect_calendar_quarterly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut year = start_date.year();
    let mut q = month_to_quarter(start_date.month());

    loop {
        let candidate = match if want_first_day {
            first_day_of_quarter(year, q)
        } else {
            last_day_of_quarter(year, q)
        } {
            Ok(date) => date,
            Err(_) => {
                // Handle potential panic from helpers if q is invalid
                eprintln!(
                    "Warning: Invalid date generated for quarter {}-Q{}",
                    year, q
                );
                break;
            }
        };

        if candidate > end_date {
            break;
        }
        if candidate >= start_date {
            result.push(candidate);
        }

        if q == 4 {
            year += 1;
            q = 1;
        } else {
            q += 1;
        }

        // Check if the *start* of the *next* quarter is already past the end date
        // to potentially break earlier.
        match first_day_of_quarter(year, q) {
            Ok(next_q_start) => {
                if next_q_start > end_date && want_first_day {
                    break;
                } // If we want start, no need to check further
                if next_q_start > end_date && !want_first_day && candidate < start_date {
                    break;
                } // If we want end and haven't found one yet, no need to check further
            }
            Err(_) => break, // Invalid next quarter, stop
        }

        // Basic loop guard
        if year > end_date.year() + 2 {
            eprintln!("Warning: Quarter loop seems excessive, breaking.");
            break;
        }
    }
    result
}

fn collect_calendar_yearly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    want_first_day: bool,
) -> Vec<NaiveDate> {
    let mut result = Vec::new();
    let mut year = start_date.year();

    while year <= end_date.year() {
        let candidate_res = if want_first_day {
            NaiveDate::from_ymd_opt(year, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(year, 12, 31)
        };

        let candidate = match candidate_res {
            Some(date) => date,
            None => {
                // Should only happen near chrono::MINYEAR/MAXYEAR
                eprintln!("Warning: Invalid date generated for year {}", year);
                year += 1; // Try next year
                continue;
            }
        };

        if candidate > end_date {
            break;
        } // Candidate past range end
        if candidate >= start_date {
            result.push(candidate);
        }

        year += 1;
    }
    result
}

// --- Internal Date Utility Functions ---

/// Given a date and a `target_weekday`, returns the date that is the first
/// `target_weekday` on or after the given date.
fn move_to_weekday_on_or_after(date: NaiveDate, target: Weekday) -> NaiveDate {
    let current_weekday_num = date.weekday().num_days_from_monday();
    let target_weekday_num = target.num_days_from_monday();
    let days_forward = (target_weekday_num + 7 - current_weekday_num) % 7;
    // Use checked_add for safety near date limits, though expect is common practice here
    date.checked_add_signed(Duration::days(days_forward as i64))
        .expect("Date calculation overflow near MAX/MIN date")
}

/// Returns the number of days in a given month and year. Returns None if month is invalid.
fn days_in_month(year: i32, month: u32) -> Option<u32> {
    if !(1..=12).contains(&month) {
        return None; // Explicitly handle invalid months
    }
    let (ny, nm) = if month == 12 {
        (year.checked_add(1)?, 1)
    } else {
        (year, month + 1)
    };
    let first_of_next = NaiveDate::from_ymd_opt(ny, nm, 1)?;
    let last_of_this = first_of_next.pred_opt()?;
    Some(last_of_this.day())
}

/// Converts a month number (1-12) to a quarter number (1-4).
/// Panics if month is invalid.
fn month_to_quarter(m: u32) -> u32 {
    assert!((1..=12).contains(&m), "Invalid month: {}", m);
    (m - 1) / 3 + 1
}

/// Returns the 1st day of the month that starts a given (year, quarter).
/// Returns Err if quarter is invalid or date calculation fails.
fn first_day_of_quarter(year: i32, quarter: u32) -> Result<NaiveDate, &'static str> {
    let month = match quarter {
        1 => 1,
        2 => 4,
        3 => 7,
        4 => 10,
        _ => return Err("Invalid quarter"),
    };
    NaiveDate::from_ymd_opt(year, month, 1).ok_or("Invalid date from quarter")
}

/// Returns the *actual* last calendar day in the given (year, quarter).
/// Returns Err if quarter is invalid or date calculation fails.
fn last_day_of_quarter(year: i32, quarter: u32) -> Result<NaiveDate, &'static str> {
    let last_month_in_quarter = match quarter {
        1 => 3,
        2 => 6,
        3 => 9,
        4 => 12,
        _ => return Err("Invalid quarter"),
    };
    let last_day =
        days_in_month(year, last_month_in_quarter).ok_or("Invalid month for quarter end")?;
    NaiveDate::from_ymd_opt(year, last_month_in_quarter, last_day)
        .ok_or("Invalid date for quarter end")
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module
    use chrono::NaiveDate;

    // Helper function to create NaiveDate instances easily in tests
    fn d(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    // --- Tests for DateFreq ---

    #[test]
    fn test_date_freq_from_str_valid() {
        assert_eq!(DateFreq::from_str("D").unwrap(), DateFreq::Daily);
        assert_eq!(DateFreq::from_str("W").unwrap(), DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("WS").unwrap(), DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("WF").unwrap(), DateFreq::WeeklyFriday);
        assert_eq!(DateFreq::from_str("M").unwrap(), DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("MS").unwrap(), DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("ME").unwrap(), DateFreq::MonthEnd);
        assert_eq!(DateFreq::from_str("Q").unwrap(), DateFreq::QuarterStart);
        assert_eq!(DateFreq::from_str("QS").unwrap(), DateFreq::QuarterStart);
        assert_eq!(DateFreq::from_str("QE").unwrap(), DateFreq::QuarterEnd);
        assert_eq!(DateFreq::from_str("Y").unwrap(), DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("A").unwrap(), DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("AS").unwrap(), DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("YS").unwrap(), DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("YE").unwrap(), DateFreq::YearEnd);
        assert_eq!(DateFreq::from_str("AE").unwrap(), DateFreq::YearEnd);
    }

    #[test]
    fn test_date_freq_from_string_valid() {
        // Test the wrapper function
        assert_eq!(
            DateFreq::from_string("D".to_string()).unwrap(),
            DateFreq::Daily
        );
        assert_eq!(
            DateFreq::from_string("YE".to_string()).unwrap(),
            DateFreq::YearEnd
        );
    }

    #[test]
    fn test_date_freq_from_str_invalid() {
        assert!(DateFreq::from_str("X").is_err());
        assert!(DateFreq::from_str(" Monthly").is_err());
        assert!(DateFreq::from_str("").is_err());
    }

    #[test]
    fn test_date_freq_from_string_invalid() {
        assert!(DateFreq::from_string("Invalid Freq".to_string()).is_err());
    }

    #[test]
    fn test_date_freq_to_string() {
        assert_eq!(DateFreq::Daily.to_string(), "D");
        assert_eq!(DateFreq::WeeklyMonday.to_string(), "W");
        assert_eq!(DateFreq::WeeklyFriday.to_string(), "WF");
        assert_eq!(DateFreq::MonthStart.to_string(), "M");
        assert_eq!(DateFreq::MonthEnd.to_string(), "ME");
        assert_eq!(DateFreq::QuarterStart.to_string(), "Q");
        assert_eq!(DateFreq::QuarterEnd.to_string(), "QE");
        assert_eq!(DateFreq::YearStart.to_string(), "Y");
        assert_eq!(DateFreq::YearEnd.to_string(), "YE");
    }

    #[test]
    fn test_date_freq_agg_type() {
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

    // --- Tests for DatesList Accessors ---

    #[test]
    fn test_dates_list_accessors() {
        let start = "2024-01-15";
        let end = "2024-02-20";
        let freq = DateFreq::WeeklyMonday;
        let dl = DatesList::new(start.to_string(), end.to_string(), freq);

        assert_eq!(dl.start_date_str(), start);
        assert_eq!(dl.end_date_str(), end);
        assert_eq!(dl.start_date().unwrap(), d(2024, 1, 15));
        assert_eq!(dl.end_date().unwrap(), d(2024, 2, 20));
        assert_eq!(dl.freq(), freq);
        assert_eq!(dl.freq_str(), "W"); // Canonical string for WeeklyMonday
    }

    #[test]
    fn test_dates_list_invalid_dates() {
        let dl_bad_start = DatesList::new(
            "2024-13-01".to_string(),
            "2024-01-10".to_string(),
            DateFreq::Daily,
        );
        assert!(dl_bad_start.start_date().is_err());
        assert!(dl_bad_start.list().is_err()); // list() should propagate parse error
        assert!(dl_bad_start.count().is_err());
        assert!(dl_bad_start.groups().is_err());

        let dl_bad_end = DatesList::new(
            "2024-01-01".to_string(),
            "invalid-date".to_string(),
            DateFreq::Daily,
        );
        assert!(dl_bad_end.end_date().is_err());
        assert!(dl_bad_end.list().is_err());
        assert!(dl_bad_end.count().is_err());
        assert!(dl_bad_end.groups().is_err());
    }

    // --- Tests for DatesList::list() and DatesList::count() ---

    #[test]
    fn test_dates_list_empty_range() {
        let dl = DatesList::new(
            "2024-01-10".to_string(),
            "2024-01-09".to_string(),
            DateFreq::Daily,
        );
        assert_eq!(dl.list().unwrap(), Vec::new());
        assert_eq!(dl.count().unwrap(), 0);
        assert_eq!(dl.groups().unwrap(), Vec::<Vec<NaiveDate>>::new()); // Ensure groups also handles empty
    }

    #[test]
    fn test_dates_list_single_day_range() {
        let dl = DatesList::new(
            "2024-01-10".to_string(),
            "2024-01-10".to_string(),
            DateFreq::Daily,
        );
        assert_eq!(dl.list().unwrap(), vec![d(2024, 1, 10)]);
        assert_eq!(dl.count().unwrap(), 1);
    }

    #[test]
    fn test_dates_list_daily() {
        let dl = DatesList::new(
            "2024-03-29".to_string(),
            "2024-04-02".to_string(),
            DateFreq::Daily,
        );
        let expected = vec![
            d(2024, 3, 29), // Fri
            d(2024, 3, 30), // Sat
            d(2024, 3, 31), // Sun
            d(2024, 4, 1),  // Mon
            d(2024, 4, 2),  // Tue
        ];
        assert_eq!(dl.list().unwrap(), expected);
        assert_eq!(dl.count().unwrap(), 5);
    }

    #[test]
    fn test_dates_list_weekly_monday() {
        // Start before first Mon, end after last Mon
        let dl1 = DatesList::new(
            "2024-01-02".to_string(),
            "2024-01-20".to_string(),
            DateFreq::WeeklyMonday,
        );
        assert_eq!(dl1.list().unwrap(), vec![d(2024, 1, 8), d(2024, 1, 15)]);
        assert_eq!(dl1.count().unwrap(), 2);

        // Start on a Mon, end on a Mon
        let dl2 = DatesList::new(
            "2024-01-08".to_string(),
            "2024-01-22".to_string(),
            DateFreq::WeeklyMonday,
        );
        assert_eq!(
            dl2.list().unwrap(),
            vec![d(2024, 1, 8), d(2024, 1, 15), d(2024, 1, 22)]
        );
        assert_eq!(dl2.count().unwrap(), 3);

        // Across year boundary
        let dl3 = DatesList::new(
            "2023-12-28".to_string(),
            "2024-01-10".to_string(),
            DateFreq::WeeklyMonday,
        );
        assert_eq!(dl3.list().unwrap(), vec![d(2024, 1, 1), d(2024, 1, 8)]);
        assert_eq!(dl3.count().unwrap(), 2);
    }

    #[test]
    fn test_dates_list_weekly_friday() {
        // Start before first Fri, end after last Fri
        let dl1 = DatesList::new(
            "2024-01-01".to_string(),
            "2024-01-20".to_string(),
            DateFreq::WeeklyFriday,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2024, 1, 5), d(2024, 1, 12), d(2024, 1, 19)]
        );
        assert_eq!(dl1.count().unwrap(), 3);

        // Start on a Fri, end on a Fri
        let dl2 = DatesList::new(
            "2024-01-12".to_string(),
            "2024-01-26".to_string(),
            DateFreq::WeeklyFriday,
        );
        assert_eq!(
            dl2.list().unwrap(),
            vec![d(2024, 1, 12), d(2024, 1, 19), d(2024, 1, 26)]
        );
        assert_eq!(dl2.count().unwrap(), 3);

        // Across year boundary
        let dl3 = DatesList::new(
            "2023-12-25".to_string(),
            "2024-01-15".to_string(),
            DateFreq::WeeklyFriday,
        );
        assert_eq!(
            dl3.list().unwrap(),
            vec![d(2023, 12, 29), d(2024, 1, 5), d(2024, 1, 12)]
        );
        assert_eq!(dl3.count().unwrap(), 3);
    }

    #[test]
    fn test_dates_list_month_start() {
        // Basic range
        let dl1 = DatesList::new(
            "2024-01-15".to_string(),
            "2024-04-10".to_string(),
            DateFreq::MonthStart,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2024, 2, 1), d(2024, 3, 1), d(2024, 4, 1)]
        );
        assert_eq!(dl1.count().unwrap(), 3);

        // Start exactly on MonthStart
        let dl2 = DatesList::new(
            "2024-02-01".to_string(),
            "2024-03-15".to_string(),
            DateFreq::MonthStart,
        );
        assert_eq!(dl2.list().unwrap(), vec![d(2024, 2, 1), d(2024, 3, 1)]);
        assert_eq!(dl2.count().unwrap(), 2);

        // Across year boundary
        let dl3 = DatesList::new(
            "2023-11-20".to_string(),
            "2024-02-10".to_string(),
            DateFreq::MonthStart,
        );
        assert_eq!(
            dl3.list().unwrap(),
            vec![d(2023, 12, 1), d(2024, 1, 1), d(2024, 2, 1)]
        );
        assert_eq!(dl3.count().unwrap(), 3);
    }

    #[test]
    fn test_dates_list_month_end() {
        // Basic range including leap year
        let dl1 = DatesList::new(
            "2024-01-15".to_string(),
            "2024-04-10".to_string(),
            DateFreq::MonthEnd,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2024, 1, 31), d(2024, 2, 29), d(2024, 3, 31)]
        ); // Feb 29 leap
        assert_eq!(dl1.count().unwrap(), 3);

        // Start exactly on MonthEnd
        let dl2 = DatesList::new(
            "2023-11-30".to_string(),
            "2024-01-15".to_string(),
            DateFreq::MonthEnd,
        );
        assert_eq!(dl2.list().unwrap(), vec![d(2023, 11, 30), d(2023, 12, 31)]);
        assert_eq!(dl2.count().unwrap(), 2);

        // Non-leap year Feb
        let dl3 = DatesList::new(
            "2023-01-20".to_string(),
            "2023-03-10".to_string(),
            DateFreq::MonthEnd,
        );
        assert_eq!(dl3.list().unwrap(), vec![d(2023, 1, 31), d(2023, 2, 28)]); // Feb 28 non-leap
        assert_eq!(dl3.count().unwrap(), 2);
    }

    #[test]
    fn test_dates_list_quarter_start() {
        // Basic range
        let dl1 = DatesList::new(
            "2024-02-15".to_string(),
            "2024-08-10".to_string(),
            DateFreq::QuarterStart,
        );
        assert_eq!(dl1.list().unwrap(), vec![d(2024, 4, 1), d(2024, 7, 1)]);
        assert_eq!(dl1.count().unwrap(), 2);

        // Start exactly on QuarterStart
        let dl2 = DatesList::new(
            "2024-01-01".to_string(),
            "2024-07-01".to_string(),
            DateFreq::QuarterStart,
        );
        assert_eq!(
            dl2.list().unwrap(),
            vec![d(2024, 1, 1), d(2024, 4, 1), d(2024, 7, 1)]
        );
        assert_eq!(dl2.count().unwrap(), 3);

        // Across year boundary
        let dl3 = DatesList::new(
            "2023-10-15".to_string(),
            "2024-05-10".to_string(),
            DateFreq::QuarterStart,
        );
        assert_eq!(dl3.list().unwrap(), vec![d(2024, 1, 1), d(2024, 4, 1)]);
        assert_eq!(dl3.count().unwrap(), 2);
    }

    #[test]
    fn test_dates_list_quarter_end() {
        // Basic range
        let dl1 = DatesList::new(
            "2024-02-15".to_string(),
            "2024-10-10".to_string(),
            DateFreq::QuarterEnd,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2024, 3, 31), d(2024, 6, 30), d(2024, 9, 30)]
        );
        assert_eq!(dl1.count().unwrap(), 3);

        // End exactly on QuarterEnd
        let dl2 = DatesList::new(
            "2024-05-01".to_string(),
            "2024-09-30".to_string(),
            DateFreq::QuarterEnd,
        );
        assert_eq!(dl2.list().unwrap(), vec![d(2024, 6, 30), d(2024, 9, 30)]);
        assert_eq!(dl2.count().unwrap(), 2);

        // Across year boundary (includes leap year effect on Mar 31 if applicable)
        let dl3 = DatesList::new(
            "2023-11-20".to_string(),
            "2024-04-10".to_string(),
            DateFreq::QuarterEnd,
        );
        assert_eq!(dl3.list().unwrap(), vec![d(2023, 12, 31), d(2024, 3, 31)]);
        assert_eq!(dl3.count().unwrap(), 2);
    }

    #[test]
    fn test_dates_list_year_start() {
        // Basic range
        let dl1 = DatesList::new(
            "2023-05-15".to_string(),
            "2026-08-10".to_string(),
            DateFreq::YearStart,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2024, 1, 1), d(2025, 1, 1), d(2026, 1, 1)]
        );
        assert_eq!(dl1.count().unwrap(), 3);

        // Start exactly on YearStart
        let dl2 = DatesList::new(
            "2024-01-01".to_string(),
            "2025-01-01".to_string(),
            DateFreq::YearStart,
        );
        assert_eq!(dl2.list().unwrap(), vec![d(2024, 1, 1), d(2025, 1, 1)]);
        assert_eq!(dl2.count().unwrap(), 2);

        // Short range within a year
        let dl3 = DatesList::new(
            "2024-02-01".to_string(),
            "2024-11-30".to_string(),
            DateFreq::YearStart,
        );
        assert_eq!(dl3.list().unwrap(), Vec::<NaiveDate>::new()); // No Jan 1st within this range
        assert_eq!(dl3.count().unwrap(), 0);
    }

    #[test]
    fn test_dates_list_year_end() {
        // Basic range
        let dl1 = DatesList::new(
            "2023-05-15".to_string(),
            "2026-08-10".to_string(),
            DateFreq::YearEnd,
        );
        assert_eq!(
            dl1.list().unwrap(),
            vec![d(2023, 12, 31), d(2024, 12, 31), d(2025, 12, 31)]
        );
        assert_eq!(dl1.count().unwrap(), 3);

        // End exactly on YearEnd
        let dl2 = DatesList::new(
            "2024-01-01".to_string(),
            "2025-12-31".to_string(),
            DateFreq::YearEnd,
        );
        assert_eq!(dl2.list().unwrap(), vec![d(2024, 12, 31), d(2025, 12, 31)]);
        assert_eq!(dl2.count().unwrap(), 2);

        // Short range within a year
        let dl3 = DatesList::new(
            "2024-02-01".to_string(),
            "2024-11-30".to_string(),
            DateFreq::YearEnd,
        );
        assert_eq!(dl3.list().unwrap(), Vec::<NaiveDate>::new()); // No Dec 31st within this range
        assert_eq!(dl3.count().unwrap(), 0);
    }

    // --- Tests for DatesList::groups() ---

    #[test]
    fn test_dates_list_groups_daily() {
        // Grouping daily just puts each date in its own group
        let dl = DatesList::new(
            "2024-01-01".to_string(),
            "2024-01-03".to_string(),
            DateFreq::Daily,
        );
        let expected = vec![
            vec![d(2024, 1, 1)],
            vec![d(2024, 1, 2)],
            vec![d(2024, 1, 3)],
        ];
        assert_eq!(dl.groups().unwrap(), expected);
    }

    #[test]
    fn test_dates_list_groups_weekly() {
        // Test grouping by ISO week, crossing month/year
        // Dates generated will be Mondays within the range
        let dl = DatesList::new(
            "2023-12-28".to_string(),
            "2024-01-16".to_string(),
            DateFreq::WeeklyMonday,
        );
        // Dates: 2024-01-01 (Week 1 2024), 2024-01-08 (Week 2 2024), 2024-01-15 (Week 3 2024)
        let expected = vec![
            vec![d(2024, 1, 1)],  // Group for Week 1 2024
            vec![d(2024, 1, 8)],  // Group for Week 2 2024
            vec![d(2024, 1, 15)], // Group for Week 3 2024
        ];
        assert_eq!(dl.groups().unwrap(), expected);

        // Test Weekly Friday grouping
        let dl_fri = DatesList::new(
            "2024-01-01".to_string(),
            "2024-01-12".to_string(),
            DateFreq::WeeklyFriday,
        );
        // Dates: 2024-01-05 (Week 1), 2024-01-12 (Week 2)
        let expected_fri = vec![
            vec![d(2024, 1, 5)],  // Group for Week 1 2024
            vec![d(2024, 1, 12)], // Group for Week 2 2024
        ];
        assert_eq!(dl_fri.groups().unwrap(), expected_fri);
    }

    #[test]
    fn test_dates_list_groups_monthly() {
        let dl = DatesList::new(
            "2023-11-15".to_string(),
            "2024-02-15".to_string(),
            DateFreq::MonthStart,
        );
        // Dates: 2023-12-01, 2024-01-01, 2024-02-01
        // Groups: (2023, 12), (2024, 1), (2024, 2)
        let expected = vec![
            vec![d(2023, 12, 1)], // Group 2023-12
            vec![d(2024, 1, 1)],  // Group 2024-01
            vec![d(2024, 2, 1)],  // Group 2024-02
        ];
        assert_eq!(dl.groups().unwrap(), expected);

        // Test Month End grouping
        let dl_me = DatesList::new(
            "2024-01-20".to_string(),
            "2024-03-10".to_string(),
            DateFreq::MonthEnd,
        );
        // Dates: 2024-01-31, 2024-02-29
        let expected_me = vec![
            vec![d(2024, 1, 31)], // Group 2024-01
            vec![d(2024, 2, 29)], // Group 2024-02
        ];
        assert_eq!(dl_me.groups().unwrap(), expected_me);
    }

    #[test]
    fn test_dates_list_groups_quarterly() {
        let dl = DatesList::new(
            "2023-08-01".to_string(),
            "2024-05-01".to_string(),
            DateFreq::QuarterStart,
        );
        // Dates: 2023-10-01 (Q4), 2024-01-01 (Q1), 2024-04-01 (Q2)
        // Groups: (2023, 4), (2024, 1), (2024, 2)
        let expected = vec![
            vec![d(2023, 10, 1)], // Group 2023-Q4
            vec![d(2024, 1, 1)],  // Group 2024-Q1
            vec![d(2024, 4, 1)],  // Group 2024-Q2
        ];
        assert_eq!(dl.groups().unwrap(), expected);

        // Test Quarter End grouping
        let dl_qe = DatesList::new(
            "2023-11-01".to_string(),
            "2024-04-15".to_string(),
            DateFreq::QuarterEnd,
        );
        // Dates: 2023-12-31 (Q4), 2024-03-31 (Q1)
        let expected_qe = vec![
            vec![d(2023, 12, 31)], // Group 2023-Q4
            vec![d(2024, 3, 31)],  // Group 2024-Q1
        ];
        assert_eq!(dl_qe.groups().unwrap(), expected_qe);
    }

    #[test]
    fn test_dates_list_groups_yearly() {
        let dl = DatesList::new(
            "2022-05-01".to_string(),
            "2024-12-31".to_string(),
            DateFreq::YearEnd,
        );
        // Dates: 2022-12-31, 2023-12-31, 2024-12-31
        // Groups: (2022), (2023), (2024)
        let expected = vec![
            vec![d(2022, 12, 31)], // Group 2022
            vec![d(2023, 12, 31)], // Group 2023
            vec![d(2024, 12, 31)], // Group 2024
        ];
        assert_eq!(dl.groups().unwrap(), expected);

        // Test Year Start grouping
        let dl_ys = DatesList::new(
            "2023-02-01".to_string(),
            "2025-01-01".to_string(),
            DateFreq::YearStart,
        );
        // Dates: 2024-01-01, 2025-01-01
        let expected_ys = vec![
            vec![d(2024, 1, 1)], // Group 2024
            vec![d(2025, 1, 1)], // Group 2025
        ];
        assert_eq!(dl_ys.groups().unwrap(), expected_ys);
    }

    // --- Tests for Utility Functions (Direct Testing) ---

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2024, 1).unwrap(), 31); // Jan
        assert_eq!(days_in_month(2024, 2).unwrap(), 29); // Feb Leap
        assert_eq!(days_in_month(2023, 2).unwrap(), 28); // Feb Non-Leap
        assert_eq!(days_in_month(2024, 4).unwrap(), 30); // Apr
        assert_eq!(days_in_month(2024, 12).unwrap(), 31); // Dec

        // Test invalid months
        assert!(days_in_month(2024, 0).is_none());
        assert!(days_in_month(2024, 13).is_none());
    }

    // Panic tests for functions with assertions/panics

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
    fn test_first_day_of_quarter() {
        assert_eq!(first_day_of_quarter(2024, 1).unwrap(), d(2024, 1, 1));
        assert_eq!(first_day_of_quarter(2024, 2).unwrap(), d(2024, 4, 1));
        assert_eq!(first_day_of_quarter(2024, 3).unwrap(), d(2024, 7, 1));
        assert_eq!(first_day_of_quarter(2024, 4).unwrap(), d(2024, 10, 1));

        assert!(first_day_of_quarter(2024, 0).is_err());
        assert!(first_day_of_quarter(2024, 5).is_err());
    }

    #[test]
    fn test_last_day_of_quarter() {
        assert_eq!(last_day_of_quarter(2024, 1).unwrap(), d(2024, 3, 31));
        assert_eq!(last_day_of_quarter(2024, 2).unwrap(), d(2024, 6, 30));
        assert_eq!(last_day_of_quarter(2024, 3).unwrap(), d(2024, 9, 30));
        assert_eq!(last_day_of_quarter(2024, 4).unwrap(), d(2024, 12, 31));
        // Check leap year effect indirectly via days_in_month tested elsewhere
        assert_eq!(last_day_of_quarter(2023, 1).unwrap(), d(2023, 3, 31)); // Non-leap doesn't affect Q1 end

        assert!(last_day_of_quarter(2024, 0).is_err());
        assert!(last_day_of_quarter(2024, 5).is_err());
    }

    #[test]
    fn test_move_to_weekday_on_or_after() {
        // Start Mon -> target Mon
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 1), Weekday::Mon),
            d(2024, 1, 1)
        );
        // Start Mon -> target Fri
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 1), Weekday::Fri),
            d(2024, 1, 5)
        );
        // Start Tue -> target Mon
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 2), Weekday::Mon),
            d(2024, 1, 8)
        );
        // Start Sat -> target Mon
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 6), Weekday::Mon),
            d(2024, 1, 8)
        );
        // Start Sun -> target Sun
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 7), Weekday::Sun),
            d(2024, 1, 7)
        );
        // Start Sun -> target Sat (next week)
        assert_eq!(
            move_to_weekday_on_or_after(d(2024, 1, 7), Weekday::Sat),
            d(2024, 1, 13)
        );
    }

    // Test potential overflow cases
    #[test]
    fn test_collect_calendar_near_max_date() {
        // Note: NaiveDate::MAX can cause issues with succ_opt/pred_opt in helpers like days_in_month
        // Let's test slightly away from the absolute max/min
        let end_date = NaiveDate::MAX.pred_opt().unwrap(); // Max - 1 day
        let start_date = end_date.pred_opt().unwrap().pred_opt().unwrap(); // Max - 3 days

        let dl = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::Daily,
        );
        let expected = vec![start_date, start_date.succ_opt().unwrap(), end_date];
        assert_eq!(dl.list().unwrap(), expected);

        // Test weekly near max date - just ensure it doesn't panic
        let dl_weekly = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::WeeklyMonday,
        );
        assert!(dl_weekly.list().is_ok());

        // Test monthly near max date
        let dl_monthly = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::MonthEnd,
        );
        assert!(dl_monthly.list().is_ok());
    }

    #[test]
    fn test_collect_calendar_near_min_date() {
        let start_date = NaiveDate::MIN.succ_opt().unwrap(); // Min + 1 day
        let end_date = start_date.succ_opt().unwrap().succ_opt().unwrap(); // Min + 3 days

        let dl = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::Daily,
        );
        let expected = vec![start_date, start_date.succ_opt().unwrap(), end_date];
        assert_eq!(dl.list().unwrap(), expected);

        // Test weekly near min date
        let dl_weekly = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::WeeklyMonday,
        );
        assert!(dl_weekly.list().is_ok());

        // Test monthly near min date
        let dl_monthly = DatesList::new(
            start_date.to_string(),
            end_date.to_string(),
            DateFreq::MonthStart,
        );
        assert!(dl_monthly.list().is_ok());
    }
} // end mod tests
