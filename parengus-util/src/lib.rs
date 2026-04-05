//! Tiny convenience utilities shared across parengus crates.
//!
//! # Dependency policy
//!
//! This crate must only ever depend on crates so ubiquitous as to be
//! practically universal (e.g. `parking_lot`, `thiserror`). No
//! domain-specific crates (graphics, audio, asset pipelines, etc.)
//! belong here.
#![cfg_attr(feature = "nightly", feature(float_algebraic))]

pub mod float;
pub mod iter;
pub mod marker;
