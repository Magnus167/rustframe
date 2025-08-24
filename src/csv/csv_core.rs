use chrono::{NaiveDate, NaiveDateTime};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Represents the target type for a CSV column.
#[derive(Debug, Clone)]
pub enum DataType {
    Int,
    Float,
    Bool,
    UInt,
    String,
    Date,
    DateTime,
}

/// Represents a value parsed from the CSV.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    UInt(u64),
    String(String),
    Date(NaiveDate),
    DateTime(NaiveDateTime),
}

/// Convenience alias for a parsed CSV record.
pub type Record = HashMap<String, Value>;

/// A simple CSV reader that reads records line by line.
pub struct CsvReader<R: BufRead> {
    reader: R,
    separators: Vec<char>,
    headers: Vec<String>,
    types: Option<HashMap<String, DataType>>,
}

/// Builder for [`CsvReader`] allowing chained configuration of headers, types, and separators.
pub struct CsvReaderBuilder<R: BufRead> {
    reader: R,
    separators: Vec<char>,
    headers: Vec<String>,
    types: Option<HashMap<String, DataType>>,
}

impl<R: BufRead> CsvReader<R> {
    /// Create a new CSV reader from a [`BufRead`] source.
    /// The first line is expected to contain headers.
    /// `separators` is a list of characters considered as field separators.
    /// `types` optionally maps column names to target data types.
    pub fn new(
        mut reader: R,
        separators: Vec<char>,
        types: Option<HashMap<String, DataType>>,
    ) -> io::Result<Self> {
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        let headers = parse_line(&first_line, &separators);
        Ok(Self {
            reader,
            separators,
            headers,
            types,
        })
    }

    /// Create a reader with default settings (comma separator, automatic typing).
    pub fn new_default(reader: R) -> io::Result<Self> {
        Self::new(reader, vec![','], None)
    }

    /// Create a reader with default separators and explicit type mapping.
    pub fn new_with_types(reader: R, types: HashMap<String, DataType>) -> io::Result<Self> {
        Self::new(reader, vec![','], Some(types))
    }

    /// Start building a reader from a source that lacks headers.
    pub fn new_with_headers(reader: R, headers: Vec<String>) -> CsvReaderBuilder<R> {
        CsvReaderBuilder {
            reader,
            separators: vec![','],
            headers,
            types: None,
        }
    }

    /// Return the headers of the CSV file.
    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Read the next record. Returns `Ok(None)` on EOF.
    pub fn read_record(&mut self) -> io::Result<Option<Record>> {
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let fields = parse_line(&line, &self.separators);
        let mut record = HashMap::new();

        for (i, header) in self.headers.iter().enumerate() {
            let field = fields.get(i).cloned().unwrap_or_default();
            let value = match &self.types {
                Some(map) => {
                    if let Some(dt) = map.get(header) {
                        parse_with_type(&field, dt)
                    } else {
                        Value::String(field)
                    }
                }
                None => parse_auto(&field),
            };
            record.insert(header.clone(), value);
        }

        Ok(Some(record))
    }
}

impl<R: BufRead> Iterator for CsvReader<R> {
    type Item = io::Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_record() {
            Ok(Some(rec)) => Some(Ok(rec)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<R: BufRead> CsvReaderBuilder<R> {
    /// Override field separators for the upcoming reader.
    pub fn separators(mut self, separators: Vec<char>) -> Self {
        self.separators = separators;
        self
    }

    /// Finalize the builder with an explicit type mapping.
    pub fn new_with_types(mut self, types: HashMap<String, DataType>) -> CsvReader<R> {
        self.types = Some(types);
        self.build()
    }

    /// Finalize the builder without specifying types.
    pub fn build(self) -> CsvReader<R> {
        CsvReader {
            reader: self.reader,
            separators: self.separators,
            headers: self.headers,
            types: self.types,
        }
    }
}

impl<R: BufRead> CsvReader<R> {
    /// Read all remaining records into a vector.
    pub fn read_all(&mut self) -> io::Result<Vec<Record>> {
        let mut records = Vec::new();
        while let Some(rec) = self.read_record()? {
            records.push(rec);
        }
        Ok(records)
    }
}

impl CsvReader<BufReader<File>> {
    /// Create a [`CsvReader`] from a file path using comma separators and
    /// automatic type detection.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustframe::csv::{CsvReader, Value};
    /// # let path = std::env::temp_dir().join("from_path_auto.csv");
    /// # std::fs::write(&path, "a,b\n1,true\n").unwrap();
    /// let mut reader = CsvReader::from_path_auto(&path).unwrap();
    /// let rec = reader.next().unwrap().unwrap();
    /// assert_eq!(rec.get("a"), Some(&Value::Int(1)));
    /// assert_eq!(rec.get("b"), Some(&Value::Bool(true)));
    /// # std::fs::remove_file(path).unwrap();
    /// ```
    pub fn from_path_auto<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        CsvReader::new_default(reader)
    }
}

/// Create an iterator over records from a file path using default settings.
pub fn reader<P: AsRef<Path>>(path: P) -> io::Result<CsvReader<BufReader<File>>> {
    reader_with(path, vec![','], None)
}

/// Create an iterator over records from a file path with custom separators and type mapping.
pub fn reader_with<P: AsRef<Path>>(
    path: P,
    separators: Vec<char>,
    types: Option<HashMap<String, DataType>>,
) -> io::Result<CsvReader<BufReader<File>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    CsvReader::new(reader, separators, types)
}

/// Read an entire CSV file into memory using default settings.
pub fn read_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<Record>> {
    read_file_with(path, vec![','], None)
}

/// Read an entire CSV file into memory with custom separators and type mapping.
pub fn read_file_with<P: AsRef<Path>>(
    path: P,
    separators: Vec<char>,
    types: Option<HashMap<String, DataType>>,
) -> io::Result<Vec<Record>> {
    let mut reader = reader_with(path, separators, types)?;
    reader.read_all()
}

fn parse_with_type(s: &str, ty: &DataType) -> Value {
    match ty {
        DataType::Int => s
            .parse::<i64>()
            .map(Value::Int)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        DataType::Float => s
            .parse::<f64>()
            .map(Value::Float)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        DataType::Bool => s
            .parse::<bool>()
            .map(Value::Bool)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        DataType::UInt => s
            .parse::<u64>()
            .map(Value::UInt)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        DataType::String => Value::String(s.to_string()),
        DataType::Date => s
            .parse::<NaiveDate>()
            .map(Value::Date)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        DataType::DateTime => NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .map(Value::DateTime)
            .unwrap_or_else(|_| Value::String(s.to_string())),
    }
}

fn parse_auto(s: &str) -> Value {
    if let Ok(i) = s.parse::<i64>() {
        Value::Int(i)
    } else if let Ok(f) = s.parse::<f64>() {
        Value::Float(f)
    } else if let Ok(b) = s.parse::<bool>() {
        Value::Bool(b)
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        Value::DateTime(dt)
    } else if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        Value::Date(d)
    } else {
        Value::String(s.to_string())
    }
}

fn parse_line(line: &str, separators: &[char]) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes: Option<char> = None;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if let Some(q) = in_quotes {
            if c == q {
                if i + 1 < chars.len() && chars[i + 1] == q {
                    current.push(q);
                    i += 1; // skip escaped quote
                } else {
                    in_quotes = None;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' || c == '\'' {
            in_quotes = Some(c);
        } else if separators.contains(&c) {
            fields.push(current.clone());
            current.clear();
        } else if c == '\r' {
            // Ignore carriage returns
        } else if c == '\n' {
            break;
        } else {
            current.push(c);
        }
        i += 1;
    }

    fields.push(current);
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, NaiveDateTime};
    use std::io::Cursor;

    #[test]
    fn test_parse_line() {
        let line = "a,'b,c',\"d\"\"e\",f";
        let fields = parse_line(line, &[',']);
        assert_eq!(fields, vec!["a", "b,c", "d\"e", "f"]);
    }

    #[test]
    fn test_reader_auto() {
        let data = "a,b,c\n1,2.5,true\n4,5.0,false\n";
        let cursor = Cursor::new(data);
        let mut reader = CsvReader::new_default(cursor).unwrap();
        let rec = reader.next().unwrap().unwrap();
        assert_eq!(rec.get("a"), Some(&Value::Int(1)));
        assert_eq!(rec.get("b"), Some(&Value::Float(2.5)));
        assert_eq!(rec.get("c"), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_reader_with_types() {
        let data = "a,b,c\n1,2,3\n";
        let cursor = Cursor::new(data);
        let mut types = HashMap::new();
        types.insert("a".to_string(), DataType::Int);
        types.insert("b".to_string(), DataType::Int);
        types.insert("c".to_string(), DataType::String);
        let mut reader = CsvReader::new_with_types(cursor, types).unwrap();
        let rec = reader.next().unwrap().unwrap();
        assert_eq!(rec.get("a"), Some(&Value::Int(1)));
        assert_eq!(rec.get("b"), Some(&Value::Int(2)));
        assert_eq!(rec.get("c"), Some(&Value::String("3".to_string())));
    }

    #[test]
    fn test_chain_headers_and_types() {
        let data = "1,2\n3,4\n";
        let cursor = Cursor::new(data);
        let headers = vec!["x".to_string(), "y".to_string()];
        let mut types = HashMap::new();
        types.insert("x".to_string(), DataType::Int);
        types.insert("y".to_string(), DataType::UInt);
        let mut reader = CsvReader::new_with_headers(cursor, headers).new_with_types(types);
        let rec = reader.next().unwrap().unwrap();
        assert_eq!(rec.get("x"), Some(&Value::Int(1)));
        assert_eq!(rec.get("y"), Some(&Value::UInt(2)));
    }

    #[test]
    fn test_date_types() {
        let data = "d,dt\n2024-01-01,2024-01-01 12:00:00\n";
        let cursor = Cursor::new(data);
        let mut types = HashMap::new();
        types.insert("d".to_string(), DataType::Date);
        types.insert("dt".to_string(), DataType::DateTime);
        let mut reader = CsvReader::new_with_types(cursor, types).unwrap();
        let rec = reader.next().unwrap().unwrap();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let datetime: NaiveDateTime = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        assert_eq!(rec.get("d"), Some(&Value::Date(date)));
        assert_eq!(rec.get("dt"), Some(&Value::DateTime(datetime)));
    }

    #[test]
    fn test_read_file_all() {
        let path = std::env::temp_dir().join("csv_full_test.csv");
        std::fs::write(&path, "a,b\n1,2\n3,4\n").unwrap();
        let records = read_file(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].get("b"), Some(&Value::Int(4)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_reader_from_path() {
        let path = std::env::temp_dir().join("csv_iter_test.csv");
        std::fs::write(&path, "a,b\n5,6\n").unwrap();
        let mut iter = reader(&path).unwrap();
        let rec = iter.next().unwrap().unwrap();
        assert_eq!(rec.get("a"), Some(&Value::Int(5)));
        assert_eq!(rec.get("b"), Some(&Value::Int(6)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_path_auto_method() {
        let path = std::env::temp_dir().join("csv_method_auto.csv");
        std::fs::write(&path, "a,b\n7,true\n").unwrap();
        let mut reader = CsvReader::from_path_auto(&path).unwrap();
        let rec = reader.next().unwrap().unwrap();
        assert_eq!(rec.get("a"), Some(&Value::Int(7)));
        assert_eq!(rec.get("b"), Some(&Value::Bool(true)));
        std::fs::remove_file(path).unwrap();
    }
}
