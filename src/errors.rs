//! Module containing custom error types for the project.

use std::error::Error;
use std::fmt;
use std::io;
use std::num::ParseIntError;

/// Error returned by the Clock when trying to exit a subroutine that has not been started.
#[derive(Debug)]
pub struct ClockError {
    pub message: String,
}

impl ClockError {
    pub fn new(message: &str) -> ClockError {
        ClockError {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ClockError {}

/// Error returned when the clq file is not formatted correctly.
#[derive(Debug)]
pub struct InvalidClqFileFormat {
    pub message: String,
}

impl InvalidClqFileFormat {
    pub fn new(message: &str) -> InvalidClqFileFormat {
        InvalidClqFileFormat {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for InvalidClqFileFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for InvalidClqFileFormat {}

impl From<io::Error> for InvalidClqFileFormat {
    fn from(err: io::Error) -> Self {
        InvalidClqFileFormat::new(&err.to_string())
    }
}

impl From<ParseIntError> for InvalidClqFileFormat {
    fn from(err: ParseIntError) -> Self {
        InvalidClqFileFormat::new(&err.to_string())
    }
}


/// Error enum containing all the possible error types that can occur while parsing a YAML file.
pub enum YamlError {
    /// Error returned when there is an error while creating / searching a file.
    IoError(String, io::Error),
    /// Error returned when an object is not found in the YAML file.
    NotFound(String, String),
    /// Error returned when an error occurs while parsing the YAML file.
    YAMLParsingError(String, serde_yaml::Error),
    /// Error returned when the YAML file is not formatted correctly.
    YAMLFormatError(String, serde_yaml::Error),
}

impl fmt::Display for YamlError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            YamlError::IoError(msg, _err) => write!(f, "{}", msg),
            YamlError::NotFound(msg, _err) => write!(f, "{}", msg),
            YamlError::YAMLParsingError(msg, _err) => write!(f, "{}.", msg),
            YamlError::YAMLFormatError(msg, _err) => write!(f, "{}.", msg),
        }
    }
}

impl fmt::Debug for YamlError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            YamlError::IoError(msg, err) => write!(f, "{}:\n {:?}", msg, err),
            YamlError::NotFound(msg, err) => write!(f, "{}:\n {:?}", msg, err),
            YamlError::YAMLParsingError(msg, err) => write!(f, "{}:\n {:?}", msg, err),
            YamlError::YAMLFormatError(msg, err) => write!(f, "{}:\n {:?}", msg, err),
        }
    }
}

impl Error for YamlError {}

impl From<serde_yaml::Error> for YamlError {
    fn from(err: serde_yaml::Error) -> Self {
        YamlError::YAMLParsingError("Error parsing YAML file".to_string(), err)
    }
}

impl From<io::Error> for YamlError {
    fn from(err: io::Error) -> Self {
        YamlError::IoError("Error while creating / opening file".to_string(), err)
    }
}

#[cfg(test)]
mod errors_test {
    use serde::de::Error;
    use crate::errors::{InvalidClqFileFormat, YamlError};

    #[test]
    fn test_clock_error() {
        use super::ClockError;
        let err = ClockError::new("Error in clock");
        assert_eq!(err.message, "Error in clock");
        assert_eq!(err.to_string(), "Error in clock");
    }


    #[test]
    fn test_invalid_clq_file_format() {
        use super::InvalidClqFileFormat;
        let err = InvalidClqFileFormat::new("Error in clq file format");
        assert_eq!(err.message, "Error in clq file format");
        assert_eq!(err.to_string(), "Error in clq file format");
    }

    #[test]
    fn test_clq_format_from_io_error() {
        let err = InvalidClqFileFormat::from(
            std::io::Error::new(std::io::ErrorKind::Other, "Error")
        );
        assert_eq!(err.message, "Error");
        assert_eq!(err.to_string(), "Error");
    }

    #[test]
    fn test_clq_format_from_parse_int() {
        // Parse integer from empty string
        let err = InvalidClqFileFormat::from(
            i32::from_str_radix("", 10).unwrap_err()
        );
        assert_eq!(err.message, "cannot parse integer from empty string");
        assert_eq!(err.to_string(), "cannot parse integer from empty string");
    }

    #[test]
    fn test_yaml_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "Error");
        let err = YamlError::IoError("Error in opening file".to_string(), io_err);
        assert_eq!(err.to_string(), "Error in opening file");
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "Error");
        assert_eq!(format!("{:?}", err), format!("Error in opening file:\n {:?}", io_err));
    }

    #[test]
    fn test_yaml_io_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "Error");
        let err = YamlError::from(io_err);
        assert_eq!(err.to_string(), "Error while creating / opening file");
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "Error");
        assert_eq!(format!("{:?}", err), format!("Error while creating / opening file:\n {:?}", io_err));
    }

    #[test]
    fn test_yaml_not_found_error() {
        let err = YamlError::NotFound("Object not found".to_string(), "Object".to_string());
        assert_eq!(err.to_string(), "Object not found");
        assert_eq!(format!("{:?}", err), "Object not found:\n \"Object\"");
    }

    #[test]
    fn test_yaml_parsing_error() {
        let yaml_err = serde_yaml::Error::custom("Error");
        let err = YamlError::YAMLParsingError("Error parsing YAML".to_string(), yaml_err);
        assert_eq!(err.to_string(), "Error parsing YAML.");
        let yaml_err = serde_yaml::Error::custom("Error");
        assert_eq!(format!("{:?}", err), format!("Error parsing YAML:\n {:?}", yaml_err));
    }

    #[test]
    fn test_yaml_parsing_error_from_yaml_error() {
        let yaml_err = serde_yaml::Error::custom("Error");
        let err = YamlError::from(yaml_err);
        assert_eq!(err.to_string(), "Error parsing YAML file.");
        let yaml_err = serde_yaml::Error::custom("Error");
        assert_eq!(format!("{:?}", err), format!("Error parsing YAML file:\n {:?}", yaml_err));
    }

    #[test]
    fn test_yaml_format_error() {
        let yaml_err = serde_yaml::Error::custom("Error");
        let err = YamlError::YAMLFormatError("Error in YAML format".to_string(), yaml_err);
        assert_eq!(err.to_string(), "Error in YAML format.");
        let yaml_err = serde_yaml::Error::custom("Error");
        assert_eq!(format!("{:?}", err), format!("Error in YAML format:\n {:?}", yaml_err));
    }

}
