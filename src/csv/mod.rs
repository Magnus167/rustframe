//! CSV handling utilities.
//!
//! The [`csv`] module offers a flexible [`CsvReader`] with automatic type
//! detection and optional builders for custom headers and types.
//!
//! # Examples
//!
//! Read from a file with auto type detection:
//!
//! ```
//! use rustframe::csv::CsvReader;
//! # let path = std::env::temp_dir().join("docs_auto.csv");
//! # std::fs::write(&path, "a,b\n1,true\n").unwrap();
//! let mut reader = CsvReader::from_path_auto(&path).unwrap();
//! for rec in reader {
//!     let rec = rec.unwrap();
//!     println!("{:?}", rec);
//! }
//! # std::fs::remove_file(path).unwrap();
//! ```
//!
//! Specify column types explicitly:
//!
//! ```
//! use rustframe::csv::{CsvReader, DataType, Value};
//! use std::collections::HashMap;
//! use std::io::Cursor;
//! let data = "a,b\n1,2\n";
//! let mut types = HashMap::new();
//! types.insert("a".into(), DataType::Int);
//! types.insert("b".into(), DataType::Float);
//! let mut reader = CsvReader::new_with_types(Cursor::new(data), types).unwrap();
//! let rec = reader.next().unwrap().unwrap();
//! assert_eq!(rec.get("b"), Some(&Value::Float(2.0)));
//! ```
//!
//! Building from custom headers and types:
//!
//! ```
//! use rustframe::csv::{CsvReader, DataType, Value};
//! use std::collections::HashMap;
//! use std::io::Cursor;
//! let data = "1,2\n";
//! let headers = vec!["x".to_string(), "y".to_string()];
//! let mut types = HashMap::new();
//! types.insert("x".into(), DataType::Int);
//! types.insert("y".into(), DataType::UInt);
//! let mut reader = CsvReader::new_with_headers(Cursor::new(data), headers).new_with_types(types);
//! let rec = reader.next().unwrap().unwrap();
//! assert_eq!(rec.get("y"), Some(&Value::UInt(2)));
//! ```
//!
//! Reading an entire file into memory:
//!
//! ```
//! use rustframe::csv::read_file;
//! # let path = std::env::temp_dir().join("docs_full.csv");
//! # std::fs::write(&path, "a,b\n1,2\n3,4\n").unwrap();
//! let records = read_file(&path).unwrap();
//! assert_eq!(records.len(), 2);
//! # std::fs::remove_file(path).unwrap();
//! ```

pub mod csv_core;

pub use csv_core::{
    CsvReader, CsvReaderBuilder, DataType, Record, Value, reader, reader_with,
    read_file, read_file_with,
};
