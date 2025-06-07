//! This module provides functionality for generating and manipulating business dates.
//! It includes the `BDatesList`, which emulates a `DateList` structure and its properties.
//! It uses `DateList` and `DateListGenerator`, adjusting the output to work on business dates.

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::error::Error;
use std::result::Result;

use crate::utils::dateutils::dates::{find_next_date, AggregationType, DateFreq, DatesGenerator};

use crate::utils::dateutils::dates;

/// Type alias for `DateFreq` to represent business date frequency.
pub type BDateFreq = DateFreq;

/// Represents a list of business dates generated between a start and end date
/// at a specified frequency. Provides methods to retrieve the full list,
/// count, or dates grouped by period.
#[derive(Debug, Clone)]
pub struct BDatesList {
    start_date_str: String,
    end_date_str: String,
    freq: DateFreq,
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
/// use rustframe::utils::{BDatesList, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let start_date = "2023-11-01".to_string(); // Wednesday
///     let end_date = "2023-11-07".to_string();   // Tuesday
///     let freq = DateFreq::Daily;
///
///     let bdates = BDatesList::new(start_date, end_date, freq);
///
///     let expected_dates = vec![
///         NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(), // Wed
///         NaiveDate::from_ymd_opt(2023, 11, 2).unwrap(), // Thu
///         NaiveDate::from_ymd_opt(2023, 11, 3).unwrap(), // Fri
///         NaiveDate::from_ymd_opt(2023, 11, 6).unwrap(), // Mon
///         NaiveDate::from_ymd_opt(2023, 11, 7).unwrap(), // Tue
///     ];
///
///     assert_eq!(bdates.list()?, expected_dates);
///     assert_eq!(bdates.count()?, 5);
///     Ok(())
/// }
/// ```
///
/// **2. Using `from_n_periods` (Start Date and Count):**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{BDatesList, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let start_date = "2024-02-28".to_string(); // Wednesday
///     let freq = DateFreq::WeeklyFriday;
///     let n_periods = 3;
///
///     let bdates = BDatesList::from_n_periods(start_date, freq, n_periods)?;
///
///     // The first Friday on or after 2024-02-28 is Mar 1.
///     // The next two Fridays are Mar 8 and Mar 15.
///     let expected_dates = vec![
///         NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
///         NaiveDate::from_ymd_opt(2024, 3, 8).unwrap(),
///         NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
///     ];
///
///     assert_eq!(bdates.list()?, expected_dates);
///     assert_eq!(bdates.count()?, 3);
///     assert_eq!(bdates.start_date_str(), "2024-02-28"); // Keeps original start string
///     assert_eq!(bdates.end_date_str(), "2024-03-15");   // End date is the last generated date
///     Ok(())
/// }
/// ```
///
/// **3. Using `groups()`:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{BDatesList, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let start_date = "2023-11-20".to_string(); // Mon, Week 47
///     let end_date = "2023-12-08".to_string();   // Fri, Week 49
///     let freq = DateFreq::WeeklyMonday;
///
///     let bdates = BDatesList::new(start_date, end_date, freq);
///
///     // Mondays in range: Nov 20, Nov 27, Dec 4
///     let groups = bdates.groups()?;
///
///     assert_eq!(groups.len(), 3); // One group per week containing a Monday
///     assert_eq!(groups[0], vec![NaiveDate::from_ymd_opt(2023, 11, 20).unwrap()]); // Week 47
///     assert_eq!(groups[1], vec![NaiveDate::from_ymd_opt(2023, 11, 27).unwrap()]); // Week 48
///     assert_eq!(groups[2], vec![NaiveDate::from_ymd_opt(2023, 12, 4).unwrap()]);  // Week 49
///     Ok(())
/// }
/// ```
impl BDatesList {
    /// Creates a new `BDatesList` instance defined by a start and end date.
    ///
    /// # Arguments
    ///
    /// * `start_date_str` - The inclusive start date as a string (e.g., "YYYY-MM-DD").
    /// * `end_date_str` - The inclusive end date as a string (e.g., "YYYY-MM-DD").
    /// * `freq` - The frequency for generating dates.
    pub fn new(start_date_str: String, end_date_str: String, freq: DateFreq) -> Self {
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
        freq: DateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if n_periods == 0 {
            return Err("n_periods must be greater than 0".into());
        }

        let start_date = NaiveDate::parse_from_str(&start_date_str, "%Y-%m-%d")?;

        // Instantiate the date generator to compute the sequence of business dates.
        let generator = BDatesGenerator::new(start_date, freq, n_periods)?;
        let dates: Vec<NaiveDate> = generator.collect();

        // Confirm that the generator returned at least one date when n_periods > 0.
        let last_date = dates
            .last()
            .ok_or("Generator failed to produce dates for the specified periods")?;

        let end_date_str = last_date.format("%Y-%m-%d").to_string();

        Ok(BDatesList {
            start_date_str,
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
        // Retrieve the list of business dates via the shared helper function.
        get_bdates_list_with_freq(&self.start_date_str, &self.end_date_str, self.freq)
    }

    /// Returns the count of business dates within the specified range and frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn count(&self) -> Result<usize, Box<dyn Error>> {
        // Compute the total number of business dates by invoking `list()` and returning its length.
        self.list().map(|list| list.len())
    }

    /// Returns a list of date lists, where each inner list contains dates
    /// belonging to the same period (determined by frequency).
    ///
    /// The outer list (groups) is sorted chronologically by period, and the
    /// inner lists (dates within each period) are also sorted.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or end date strings cannot be parsed.
    pub fn groups(&self) -> Result<Vec<Vec<NaiveDate>>, Box<dyn Error>> {
        let dates = self.list()?;
        dates::group_dates_helper(dates, self.freq)
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

// Business date iterator: generates a sequence of business dates for a given frequency and period count.

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
/// use rustframe::utils::{BDatesGenerator, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let start = NaiveDate::from_ymd_opt(2023, 12, 28).unwrap(); // Thursday
///     let freq = DateFreq::MonthEnd;
///     let n_periods = 4; // Dec '23, Jan '24, Feb '24, Mar '24
///
///     let mut generator = BDatesGenerator::new(start, freq, n_periods)?;
///
///     // First month-end on or after 2023-12-28 is 2023-12-29
///     assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2023, 12, 29).unwrap()));
///     assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 1, 31).unwrap()));
///     assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 2, 29).unwrap())); // Leap year
///     assert_eq!(generator.next(), Some(NaiveDate::from_ymd_opt(2024, 3, 29).unwrap())); // Mar 31 is Sun
///     assert_eq!(generator.next(), None); // Exhausted
///     Ok(())
/// }
/// ```
///
/// **2. Collecting into a Vec:**
///
/// ```rust
/// use chrono::NaiveDate;
/// use std::error::Error;
/// use rustframe::utils::{BDatesGenerator, DateFreq};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let start = NaiveDate::from_ymd_opt(2024, 4, 29).unwrap(); // Monday
///     let freq = DateFreq::Daily;
///     let n_periods = 5;
///
///     let generator = BDatesGenerator::new(start, freq, n_periods)?;
///     let dates: Vec<NaiveDate> = generator.collect();
///
///     let expected_dates = vec![
///         NaiveDate::from_ymd_opt(2024, 4, 29).unwrap(), // Mon
///         NaiveDate::from_ymd_opt(2024, 4, 30).unwrap(), // Tue
///         NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),  // Wed
///         NaiveDate::from_ymd_opt(2024, 5, 2).unwrap(),  // Thu
///         NaiveDate::from_ymd_opt(2024, 5, 3).unwrap(),  // Fri
///     ];
///
///     assert_eq!(dates, expected_dates);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct BDatesGenerator {
    dates_generator: DatesGenerator,
    start_date: NaiveDate,
    freq: DateFreq,
    periods_remaining: usize,
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
        freq: DateFreq,
        n_periods: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // over-estimate the number of periods
        let adj_n_periods = match freq {
            DateFreq::Daily => n_periods + 5,
            DateFreq::WeeklyMonday
            | DateFreq::WeeklyFriday
            | DateFreq::MonthStart
            | DateFreq::MonthEnd
            | DateFreq::QuarterStart
            | DateFreq::QuarterEnd
            | DateFreq::YearStart
            | DateFreq::YearEnd => n_periods + 2,
        };

        let dates_generator = DatesGenerator::new(start_date, freq, adj_n_periods)?;

        Ok(BDatesGenerator {
            dates_generator,
            start_date,
            freq,
            periods_remaining: n_periods,
        })
    }
}

impl Iterator for BDatesGenerator {
    type Item = NaiveDate;

    /// Returns the next business date in the sequence, or `None` if the specified
    /// number of periods has been generated.
    fn next(&mut self) -> Option<Self::Item> {
        // Terminate if no periods remain or no initial date is set.
        if self.periods_remaining == 0 {
            return None;
        }

        // get the next date from the generator
        let next_date = self.dates_generator.next()?;

        let next_date = match self.freq {
            DateFreq::Daily => {
                let mut new_candidate = next_date.clone();
                while !is_business_date(new_candidate) {
                    new_candidate = self.dates_generator.next()?;
                }
                new_candidate
            }

            DateFreq::WeeklyMonday | DateFreq::WeeklyFriday => next_date,
            DateFreq::MonthEnd | DateFreq::QuarterEnd | DateFreq::YearEnd => {
                let adjusted_date = iter_reverse_till_bdate(next_date);
                if self.start_date > adjusted_date {
                    // Skip this iteration if the adjusted date is before the start date.
                    return self.next();
                }
                adjusted_date
            }
            DateFreq::MonthStart | DateFreq::QuarterStart | DateFreq::YearStart => {
                // Adjust to the first business date of the month, quarter, or year.
                iter_till_bdate(next_date)
            }
        };
        // Decrement the remaining periods.
        self.periods_remaining -= 1;
        Some(next_date)
    }
}

/// Check if the date is a weekend (Saturday or Sunday).
pub fn is_business_date(date: NaiveDate) -> bool {
    match date.weekday() {
        Weekday::Sat | Weekday::Sun => false,
        _ => true,
    }
}

pub fn find_next_bdate(date: NaiveDate, freq: DateFreq) -> NaiveDate {
    let next_date: NaiveDate = find_next_date(date, freq).unwrap();
    let next_date = iter_till_bdate(next_date);
    next_date
}

pub fn find_first_bdate_on_or_after(date: NaiveDate, freq: DateFreq) -> NaiveDate {
    // Find the first business date on or after the given date.
    let first_date = dates::find_first_date_on_or_after(date, freq).unwrap();
    let first_date = iter_till_bdate_by_freq(first_date, freq);
    // let first_date = iter_till_bdate(first_date);

    first_date
}

/// Iterate forwards or backwards (depending on the frequency)
/// until a business date is found.
fn iter_till_bdate_by_freq(date: NaiveDate, freq: DateFreq) -> NaiveDate {
    let agg_type = freq.agg_type();
    let dur = match agg_type {
        AggregationType::Start => Duration::days(1),
        AggregationType::End => Duration::days(-1),
    };
    let mut current_date = date;
    while !is_business_date(current_date) {
        current_date = current_date + dur;
    }
    current_date
}

/// Increment day-by-day until a business date is found.
fn iter_till_bdate(date: NaiveDate) -> NaiveDate {
    let mut current_date = date;
    while !is_business_date(current_date) {
        current_date = current_date + Duration::days(1);
    }
    current_date
}

/// Increment day-by-day until a business date is found.
fn iter_reverse_till_bdate(date: NaiveDate) -> NaiveDate {
    let mut current_date = date;
    while !is_business_date(current_date) {
        current_date = current_date - Duration::days(1);
    }
    current_date
}

/// Helper function to get a list of business dates based on the frequency.
pub fn get_bdates_list_with_freq(
    start_date_str: &str,
    end_date_str: &str,
    freq: DateFreq,
) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
    // Generate the list of business dates using the shared logic.

    let start_date = NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d")?;
    let end_date = NaiveDate::parse_from_str(end_date_str, "%Y-%m-%d")?;

    let mut dates = dates::get_dates_list_with_freq_from_naive_date(start_date, end_date, freq)?;

    match freq {
        DateFreq::Daily => {
            dates.retain(|date| is_business_date(*date));
        }
        DateFreq::WeeklyMonday | DateFreq::WeeklyFriday => {
            // No logic needed (or possible?)
        }
        _ => {
            dates.iter_mut().for_each(|date| {
                *date = iter_till_bdate_by_freq(*date, freq);
            });
        }
    }

    Ok(dates)
}

// --- Example Usage and Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::str::FromStr;

    // Helper to create a NaiveDate for tests, handling the expect for fixed dates.
    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("Invalid date in test setup")
    }

    // --- DateFreq Tests ---

    #[test]
    fn test_date_freq_from_str() -> Result<(), Box<dyn Error>> {
        assert_eq!(DateFreq::from_str("D")?, DateFreq::Daily);
        assert_eq!("D".parse::<DateFreq>()?, DateFreq::Daily); // Test FromStr impl
        assert_eq!(DateFreq::from_str("W")?, DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("M")?, DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("Q")?, DateFreq::QuarterStart);

        // Test YearStart codes and aliases (Y, A, AS, YS)
        assert_eq!(DateFreq::from_str("Y")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("A")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("AS")?, DateFreq::YearStart);
        assert_eq!(DateFreq::from_str("YS")?, DateFreq::YearStart);
        assert_eq!("Y".parse::<DateFreq>()?, DateFreq::YearStart); // Test FromStr impl

        assert_eq!(DateFreq::from_str("ME")?, DateFreq::MonthEnd);
        assert_eq!(DateFreq::from_str("QE")?, DateFreq::QuarterEnd);
        assert_eq!(DateFreq::from_str("WF")?, DateFreq::WeeklyFriday);
        assert_eq!("WF".parse::<DateFreq>()?, DateFreq::WeeklyFriday); // Test FromStr impl

        // Test YearEnd codes and aliases (YE, AE)
        assert_eq!(DateFreq::from_str("YE")?, DateFreq::YearEnd);
        assert_eq!(DateFreq::from_str("AE")?, DateFreq::YearEnd);

        // Test aliases for other frequencies
        assert_eq!(DateFreq::from_str("WS")?, DateFreq::WeeklyMonday);
        assert_eq!(DateFreq::from_str("MS")?, DateFreq::MonthStart);
        assert_eq!(DateFreq::from_str("QS")?, DateFreq::QuarterStart);

        // Test invalid string
        assert!(DateFreq::from_str("INVALID").is_err());
        assert!("INVALID".parse::<DateFreq>().is_err()); // Test FromStr impl
        let err = DateFreq::from_str("INVALID").unwrap_err();
        assert_eq!(err.to_string(), "Invalid frequency specified: INVALID");

        Ok(())
    }

    #[test]
    fn test_date_freq_to_string() {
        assert_eq!(DateFreq::Daily.to_string(), "D");
        assert_eq!(DateFreq::WeeklyMonday.to_string(), "W");
        assert_eq!(DateFreq::MonthStart.to_string(), "M");
        assert_eq!(DateFreq::QuarterStart.to_string(), "Q");
        assert_eq!(DateFreq::YearStart.to_string(), "Y"); // Assert "Y"
        assert_eq!(DateFreq::MonthEnd.to_string(), "ME");
        assert_eq!(DateFreq::QuarterEnd.to_string(), "QE");
        assert_eq!(DateFreq::WeeklyFriday.to_string(), "WF");
        assert_eq!(DateFreq::YearEnd.to_string(), "YE");
    }

    #[test]
    fn test_date_freq_from_string() -> Result<(), Box<dyn Error>> {
        assert_eq!(DateFreq::from_string("D".to_string())?, DateFreq::Daily);
        assert!(DateFreq::from_string("INVALID".to_string()).is_err());
        Ok(())
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

    // --- BDatesList Property Tests ---

    #[test]
    fn test_bdates_list_properties_new() -> Result<(), Box<dyn Error>> {
        let start_str = "2023-01-01".to_string();
        let end_str = "2023-12-31".to_string();
        let freq = DateFreq::QuarterEnd;
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
        let freq = DateFreq::Daily;
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
        let freq = DateFreq::Daily;
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
        let freq = DateFreq::Daily;
        let n_periods = 5;
        let result = BDatesList::from_n_periods(start_str.clone(), freq, n_periods);
        assert!(result.is_err());
        // Error comes from NaiveDate::parse_from_str
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("input contains invalid characters"));
    }

    #[test]
    fn test_bdates_list_invalid_date_string_new() {
        let dates_list_start_invalid = BDatesList::new(
            "invalid-date".to_string(),
            "2023-12-31".to_string(),
            DateFreq::Daily,
        );
        assert!(dates_list_start_invalid.list().is_err());
        assert!(dates_list_start_invalid.count().is_err());
        assert!(dates_list_start_invalid.groups().is_err());
        assert!(dates_list_start_invalid.start_date().is_err());
        assert!(dates_list_start_invalid.end_date().is_ok()); // End date is valid

        let dates_list_end_invalid = BDatesList::new(
            "2023-01-01".to_string(),
            "invalid-date".to_string(),
            DateFreq::Daily,
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
            DateFreq::QuarterEnd,
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
            DateFreq::WeeklyMonday,
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
            DateFreq::Daily,
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
            DateFreq::Daily,
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
            DateFreq::MonthEnd,
        );
        assert_eq!(dates_list.count()?, 12, "{:?}", dates_list.list()); // 12 month ends in 2023

        let dates_list_weekly = BDatesList::new(
            "2023-11-01".to_string(), // Wed
            "2023-11-30".to_string(), // Thu
            DateFreq::WeeklyFriday,
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
            DateFreq::YearStart,
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
            DateFreq::MonthStart,
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
            DateFreq::WeeklyFriday,
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
            DateFreq::MonthEnd,
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
            DateFreq::Daily,
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
            DateFreq::WeeklyFriday,
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
            DateFreq::QuarterStart,
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
            DateFreq::YearEnd,
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
            DateFreq::Daily,
        );
        let groups = dates_list.groups()?;
        assert!(groups.is_empty());

        Ok(())
    }

    // --- Tests for BDatesGenerator ---

    #[test]
    fn test_generator_new_zero_periods() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 1, 1);
        let freq = DateFreq::Daily;
        let n_periods = 0;
        let mut generator = BDatesGenerator::new(start_date, freq, n_periods)?;
        assert_eq!(generator.next(), None); // Should be immediately exhausted
        Ok(())
    }

    #[test]
    fn test_generator_daily() -> Result<(), Box<dyn Error>> {
        let start_date = date(2023, 11, 10); // Friday
        let freq = DateFreq::Daily;
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
        let freq = DateFreq::WeeklyMonday;
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
        let freq = DateFreq::WeeklyFriday;
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
        let freq = DateFreq::MonthStart;
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
        let freq = DateFreq::MonthEnd;
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
        let freq = DateFreq::QuarterStart;
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
        let freq = DateFreq::QuarterEnd;
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
        let freq = DateFreq::YearStart;
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
        let freq = DateFreq::YearEnd;
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
        let freq = DateFreq::Daily;
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
}
