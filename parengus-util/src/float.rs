//! Algebraic and deterministic float wrappers.

pub use algebraic_f32::AlgebraicF32;
pub use algebraic_f64::AlgebraicF64;
pub use deterministic_f32::DeterministicF32;
pub use deterministic_f64::DeterministicF64;
mod algebraic_f32 {

    use num_traits::{
        Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
    };
    use std::{
        num::FpCategory,
        ops::{Add, Deref, DerefMut, Div, Mul, Neg, Rem, Sub},
    };
    /// A wrapper around `f32` whose arithmetic operators use the `algebraic_*`
    /// intrinsics when available (nightly, `float_algebraic` feature), enabling
    /// reassociation, contraction, and other algebraic optimisations on a
    /// per-operation basis without enabling `-ffast-math` crate-wide.
    ///
    /// On stable the operators fall back to ordinary `f32` arithmetic with
    /// identical semantics. Once `float_algebraic` stabilises, the `nightly`
    /// feature flag and the cfg-gated fallback impls will need to be removed
    /// to unconditionally use the algebraic ops.
    ///
    /// Deref to `f32` for access to all other `f32` methods.
    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    #[repr(transparent)]
    pub struct AlgebraicF32(pub f32);

    impl Deref for AlgebraicF32 {
        type Target = f32;
        fn deref(&self) -> &f32 {
            &self.0
        }
    }

    impl DerefMut for AlgebraicF32 {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl From<f32> for AlgebraicF32 {
        fn from(v: f32) -> Self {
            Self(v)
        }
    }

    impl From<AlgebraicF32> for f32 {
        fn from(v: AlgebraicF32) -> f32 {
            v.0
        }
    }

    impl From<AlgebraicF32> for f64 {
        fn from(v: AlgebraicF32) -> f64 {
            v.0.into()
        }
    }

    // Arithmetic — algebraic intrinsics on nightly, plain ops on stable

    #[cfg(feature = "nightly")]
    impl Add for AlgebraicF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(f32::algebraic_add(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Add for AlgebraicF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Sub for AlgebraicF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(f32::algebraic_sub(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Sub for AlgebraicF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Mul for AlgebraicF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(f32::algebraic_mul(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Mul for AlgebraicF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Div for AlgebraicF32 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(f32::algebraic_div(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Div for AlgebraicF32 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(self.0 / rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Neg for AlgebraicF32 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(f32::algebraic_neg(self.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Neg for AlgebraicF32 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    #[cfg(feature = "nightly")]
    impl Rem for AlgebraicF32 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self {
            Self(f32::algebraic_rem(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Rem for AlgebraicF32 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            Self(self.0 % rhs.0)
        }
    }

    // Num traits:
    impl Zero for AlgebraicF32 {
        fn zero() -> Self {
            Self(0.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl One for AlgebraicF32 {
        fn one() -> Self {
            Self(1.0)
        }

        fn is_one(&self) -> bool {
            self.0 == 1.0
        }
    }

    impl ToPrimitive for AlgebraicF32 {
        fn to_i64(&self) -> Option<i64> {
            Some(self.0 as i64)
        }

        fn to_u64(&self) -> Option<u64> {
            Some(self.0 as u64)
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0 as f64)
        }

        fn to_f32(&self) -> Option<f32> {
            Some(self.0)
        }
    }

    impl FromPrimitive for AlgebraicF32 {
        fn from_i64(n: i64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_u64(n: u64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_f64(n: f64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_f32(n: f32) -> Option<Self> {
            Some(Self(n))
        }
    }

    impl NumCast for AlgebraicF32 {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f32().map(Self)
        }
    }

    impl Num for AlgebraicF32 {
        type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
        fn from_str_radix(
            src: &str,
            radix: u32,
        ) -> Result<Self, Self::FromStrRadixErr> {
            <f32 as Num>::from_str_radix(src, radix).map(Self)
        }
    }

    impl Signed for AlgebraicF32 {
        fn abs(&self) -> Self {
            Self(self.0.abs())
        }

        fn abs_sub(&self, other: &Self) -> Self {
            if self.0 <= other.0 {
                Self(0.0)
            } else {
                Self(self.0 - other.0)
            }
        }

        fn signum(&self) -> Self {
            Self(self.0.signum())
        }

        fn is_positive(&self) -> bool {
            self.0.is_sign_positive()
        }

        fn is_negative(&self) -> bool {
            self.0.is_sign_negative()
        }
    }

    impl Float for AlgebraicF32 {
        fn nan() -> Self {
            Self(f32::NAN)
        }
        fn infinity() -> Self {
            Self(f32::INFINITY)
        }
        fn neg_infinity() -> Self {
            Self(f32::NEG_INFINITY)
        }
        fn neg_zero() -> Self {
            Self(-0.0)
        }
        fn min_value() -> Self {
            Self(f32::MIN)
        }
        fn min_positive_value() -> Self {
            Self(f32::MIN_POSITIVE)
        }
        fn epsilon() -> Self {
            Self(f32::EPSILON)
        }
        fn max_value() -> Self {
            Self(f32::MAX)
        }

        fn is_nan(self) -> bool {
            self.0.is_nan()
        }
        fn is_infinite(self) -> bool {
            self.0.is_infinite()
        }
        fn is_finite(self) -> bool {
            self.0.is_finite()
        }
        fn is_normal(self) -> bool {
            self.0.is_normal()
        }
        fn classify(self) -> FpCategory {
            self.0.classify()
        }

        fn floor(self) -> Self {
            Self(self.0.floor())
        }
        fn ceil(self) -> Self {
            Self(self.0.ceil())
        }
        fn round(self) -> Self {
            Self(self.0.round())
        }
        fn trunc(self) -> Self {
            Self(self.0.trunc())
        }
        fn fract(self) -> Self {
            Self(self.0.fract())
        }
        fn abs(self) -> Self {
            Self(self.0.abs())
        }
        fn signum(self) -> Self {
            Self(self.0.signum())
        }
        fn is_sign_positive(self) -> bool {
            self.0.is_sign_positive()
        }
        fn is_sign_negative(self) -> bool {
            self.0.is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            Self(self.0.mul_add(a.0, b.0))
        }
        fn recip(self) -> Self {
            Self(self.0.recip())
        }
        fn powi(self, n: i32) -> Self {
            Self(self.0.powi(n))
        }
        fn powf(self, n: Self) -> Self {
            Self(self.0.powf(n.0))
        }
        fn sqrt(self) -> Self {
            Self(self.0.sqrt())
        }
        fn exp(self) -> Self {
            Self(self.0.exp())
        }
        fn exp2(self) -> Self {
            Self(self.0.exp2())
        }
        fn ln(self) -> Self {
            Self(self.0.ln())
        }
        fn log(self, base: Self) -> Self {
            Self(self.0.log(base.0))
        }
        fn log2(self) -> Self {
            Self(self.0.log2())
        }
        fn log10(self) -> Self {
            Self(self.0.log10())
        }
        fn to_degrees(self) -> Self {
            Self(self.0.to_degrees())
        }
        fn to_radians(self) -> Self {
            Self(self.0.to_radians())
        }
        fn max(self, other: Self) -> Self {
            Self(self.0.max(other.0))
        }
        fn min(self, other: Self) -> Self {
            Self(self.0.min(other.0))
        }
        fn abs_sub(self, other: Self) -> Self {
            if self.0 <= other.0 {
                Self(0.0)
            } else {
                Self(self.0 - other.0)
            }
        }
        fn cbrt(self) -> Self {
            Self(self.0.cbrt())
        }
        fn hypot(self, other: Self) -> Self {
            Self(self.0.hypot(other.0))
        }

        fn sin(self) -> Self {
            Self(self.0.sin())
        }
        fn cos(self) -> Self {
            Self(self.0.cos())
        }
        fn tan(self) -> Self {
            Self(self.0.tan())
        }
        fn asin(self) -> Self {
            Self(self.0.asin())
        }
        fn acos(self) -> Self {
            Self(self.0.acos())
        }
        fn atan(self) -> Self {
            Self(self.0.atan())
        }
        fn atan2(self, other: Self) -> Self {
            Self(self.0.atan2(other.0))
        }
        fn sin_cos(self) -> (Self, Self) {
            let (s, c) = self.0.sin_cos();
            (Self(s), Self(c))
        }

        fn exp_m1(self) -> Self {
            Self(self.0.exp_m1())
        }
        fn ln_1p(self) -> Self {
            Self(self.0.ln_1p())
        }
        fn sinh(self) -> Self {
            Self(self.0.sinh())
        }
        fn cosh(self) -> Self {
            Self(self.0.cosh())
        }
        fn tanh(self) -> Self {
            Self(self.0.tanh())
        }
        fn asinh(self) -> Self {
            Self(self.0.asinh())
        }
        fn acosh(self) -> Self {
            Self(self.0.acosh())
        }
        fn atanh(self) -> Self {
            Self(self.0.atanh())
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            let bits: u32 = self.0.to_bits();
            let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
            let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
            let mantissa = if exponent == 0 {
                (bits & 0x7fffff) << 1
            } else {
                (bits & 0x7fffff) | 0x800000
            };
            exponent -= 127 + 23;
            (mantissa as u64, exponent, sign)
        }

        fn copysign(self, sign: Self) -> Self {
            Self(self.0.copysign(sign.0))
        }
    }
}

mod algebraic_f64 {

    use super::algebraic_f32::AlgebraicF32;
    use core::num::FpCategory;
    use core::ops::Rem;
    use num_traits::{
        Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
    };
    use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub};

    /// A wrapper around `f64` whose arithmetic operators use the `algebraic_*`
    /// intrinsics when available (nightly, `float_algebraic` feature), enabling
    /// reassociation, contraction, and other algebraic optimisations on a
    /// per-operation basis without enabling `-ffast-math` crate-wide.
    ///
    /// On stable the operators fall back to ordinary `f64` arithmetic with
    /// identical semantics; the upgrade to algebraic ops is a simple adjustment
    /// once the features stabilise
    ///
    /// Deref to `f64` for access to all other `f64` methods.
    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    #[repr(transparent)]
    pub struct AlgebraicF64(pub f64);

    impl Deref for AlgebraicF64 {
        type Target = f64;
        fn deref(&self) -> &f64 {
            &self.0
        }
    }

    impl DerefMut for AlgebraicF64 {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl From<f64> for AlgebraicF64 {
        fn from(v: f64) -> Self {
            Self(v)
        }
    }

    impl From<AlgebraicF32> for AlgebraicF64 {
        fn from(v: AlgebraicF32) -> Self {
            Self(v.0.into())
        }
    }

    impl From<f32> for AlgebraicF64 {
        fn from(v: f32) -> Self {
            Self(v.into())
        }
    }

    impl From<AlgebraicF64> for f64 {
        fn from(v: AlgebraicF64) -> f64 {
            v.0
        }
    }

    #[cfg(feature = "nightly")]
    impl Add for AlgebraicF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(f64::algebraic_add(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Add for AlgebraicF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Sub for AlgebraicF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(f64::algebraic_sub(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Sub for AlgebraicF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Mul for AlgebraicF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(f64::algebraic_mul(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Mul for AlgebraicF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Div for AlgebraicF64 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(f64::algebraic_div(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Div for AlgebraicF64 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(self.0 / rhs.0)
        }
    }

    #[cfg(feature = "nightly")]
    impl Neg for AlgebraicF64 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(f64::algebraic_neg(self.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Neg for AlgebraicF64 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    #[cfg(feature = "nightly")]
    impl Rem for AlgebraicF64 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            Self(f64::algebraic_rem(self.0, rhs.0))
        }
    }
    #[cfg(not(feature = "nightly"))]
    impl Rem for AlgebraicF64 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            Self(self.0 % rhs.0)
        }
    }

    // Num traits:
    impl Zero for AlgebraicF64 {
        fn zero() -> Self {
            Self(0.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl One for AlgebraicF64 {
        fn one() -> Self {
            Self(1.0)
        }

        fn is_one(&self) -> bool {
            self.0 == 1.0
        }
    }

    impl ToPrimitive for AlgebraicF64 {
        fn to_i64(&self) -> Option<i64> {
            Some(self.0 as i64)
        }

        fn to_u64(&self) -> Option<u64> {
            Some(self.0 as u64)
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0)
        }

        fn to_f32(&self) -> Option<f32> {
            Some(self.0 as f32)
        }
    }

    impl FromPrimitive for AlgebraicF64 {
        fn from_i64(n: i64) -> Option<Self> {
            Some(Self(n as f64))
        }

        fn from_u64(n: u64) -> Option<Self> {
            Some(Self(n as f64))
        }

        fn from_f64(n: f64) -> Option<Self> {
            Some(Self(n))
        }

        fn from_f32(n: f32) -> Option<Self> {
            Some(Self(n.into()))
        }
    }

    impl NumCast for AlgebraicF64 {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f64().map(Self)
        }
    }

    impl Num for AlgebraicF64 {
        type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
        fn from_str_radix(
            src: &str,
            radix: u32,
        ) -> Result<Self, Self::FromStrRadixErr> {
            <f64 as Num>::from_str_radix(src, radix).map(Self)
        }
    }

    impl Signed for AlgebraicF64 {
        fn abs(&self) -> Self {
            Self(self.0.abs())
        }

        fn abs_sub(&self, other: &Self) -> Self {
            if self.0 <= other.0 {
                Self(0.0)
            } else {
                Self(self.0 - other.0)
            }
        }

        fn signum(&self) -> Self {
            Self(self.0.signum())
        }

        fn is_positive(&self) -> bool {
            self.0.is_sign_positive()
        }

        fn is_negative(&self) -> bool {
            self.0.is_sign_negative()
        }
    }

    impl Float for AlgebraicF64 {
        fn nan() -> Self {
            Self(f64::NAN)
        }
        fn infinity() -> Self {
            Self(f64::INFINITY)
        }
        fn neg_infinity() -> Self {
            Self(f64::NEG_INFINITY)
        }
        fn neg_zero() -> Self {
            Self(-0.0)
        }
        fn min_value() -> Self {
            Self(f64::MIN)
        }
        fn min_positive_value() -> Self {
            Self(f64::MIN_POSITIVE)
        }
        fn epsilon() -> Self {
            Self(f64::EPSILON)
        }
        fn max_value() -> Self {
            Self(f64::MAX)
        }

        fn is_nan(self) -> bool {
            self.0.is_nan()
        }
        fn is_infinite(self) -> bool {
            self.0.is_infinite()
        }
        fn is_finite(self) -> bool {
            self.0.is_finite()
        }
        fn is_normal(self) -> bool {
            self.0.is_normal()
        }
        fn classify(self) -> FpCategory {
            self.0.classify()
        }

        fn floor(self) -> Self {
            Self(self.0.floor())
        }
        fn ceil(self) -> Self {
            Self(self.0.ceil())
        }
        fn round(self) -> Self {
            Self(self.0.round())
        }
        fn trunc(self) -> Self {
            Self(self.0.trunc())
        }
        fn fract(self) -> Self {
            Self(self.0.fract())
        }
        fn abs(self) -> Self {
            Self(self.0.abs())
        }
        fn signum(self) -> Self {
            Self(self.0.signum())
        }
        fn is_sign_positive(self) -> bool {
            self.0.is_sign_positive()
        }
        fn is_sign_negative(self) -> bool {
            self.0.is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            Self(self.0.mul_add(a.0, b.0))
        }
        fn recip(self) -> Self {
            Self(self.0.recip())
        }
        fn powi(self, n: i32) -> Self {
            Self(self.0.powi(n))
        }
        fn powf(self, n: Self) -> Self {
            Self(self.0.powf(n.0))
        }
        fn sqrt(self) -> Self {
            Self(self.0.sqrt())
        }
        fn exp(self) -> Self {
            Self(self.0.exp())
        }
        fn exp2(self) -> Self {
            Self(self.0.exp2())
        }
        fn ln(self) -> Self {
            Self(self.0.ln())
        }
        fn log(self, base: Self) -> Self {
            Self(self.0.log(base.0))
        }
        fn log2(self) -> Self {
            Self(self.0.log2())
        }
        fn log10(self) -> Self {
            Self(self.0.log10())
        }
        fn to_degrees(self) -> Self {
            Self(self.0.to_degrees())
        }
        fn to_radians(self) -> Self {
            Self(self.0.to_radians())
        }
        fn max(self, other: Self) -> Self {
            Self(self.0.max(other.0))
        }
        fn min(self, other: Self) -> Self {
            Self(self.0.min(other.0))
        }
        fn abs_sub(self, other: Self) -> Self {
            if self.0 <= other.0 {
                Self(0.0)
            } else {
                Self(self.0 - other.0)
            }
        }
        fn cbrt(self) -> Self {
            Self(self.0.cbrt())
        }
        fn hypot(self, other: Self) -> Self {
            Self(self.0.hypot(other.0))
        }

        fn sin(self) -> Self {
            Self(self.0.sin())
        }
        fn cos(self) -> Self {
            Self(self.0.cos())
        }
        fn tan(self) -> Self {
            Self(self.0.tan())
        }
        fn asin(self) -> Self {
            Self(self.0.asin())
        }
        fn acos(self) -> Self {
            Self(self.0.acos())
        }
        fn atan(self) -> Self {
            Self(self.0.atan())
        }
        fn atan2(self, other: Self) -> Self {
            Self(self.0.atan2(other.0))
        }
        fn sin_cos(self) -> (Self, Self) {
            let (s, c) = self.0.sin_cos();
            (Self(s), Self(c))
        }

        fn exp_m1(self) -> Self {
            Self(self.0.exp_m1())
        }
        fn ln_1p(self) -> Self {
            Self(self.0.ln_1p())
        }
        fn sinh(self) -> Self {
            Self(self.0.sinh())
        }
        fn cosh(self) -> Self {
            Self(self.0.cosh())
        }
        fn tanh(self) -> Self {
            Self(self.0.tanh())
        }
        fn asinh(self) -> Self {
            Self(self.0.asinh())
        }
        fn acosh(self) -> Self {
            Self(self.0.acosh())
        }
        fn atanh(self) -> Self {
            Self(self.0.atanh())
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            let bits: u64 = self.0.to_bits();
            let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
            let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
            let mantissa = if exponent == 0 {
                (bits & 0xfffffffffffff) << 1
            } else {
                (bits & 0xfffffffffffff) | 0x10000000000000
            };
            exponent -= 1023 + 52;
            (mantissa, exponent, sign)
        }

        fn copysign(self, sign: Self) -> Self {
            Self(self.0.copysign(sign.0))
        }
    }
}
mod deterministic_f32 {
    use core::num::FpCategory;
    use core::ops::Rem;
    use num_traits::{
        Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
    };
    use std::ops::{Add, Div, Mul, Neg, Sub};

    /// A wrapper around `f32` that routes all transcendental operations through
    /// [`libm`], producing bit-identical results on every platform.
    ///
    /// Standard-library transcendentals (`f32::sin`, etc.) may use hardware
    /// instructions whose precision varies by CPU and compiler. This type delegates
    /// to `libm`'s portable pure-Rust implementations instead, so results are
    /// reproducible across machines — useful for deterministic simulation,
    /// serialised state, or cross-platform tests. There are errors in here. If you
    /// want a specific level of precision, it would be better to use a library that
    /// either specifically documents their ULP or
    ///
    /// Arithmetic operators (`+`, `-`, `*`, `/`) use ordinary IEEE 754 operations,
    /// which are already correctly-rounded and cross-platform identical.
    ///
    /// **No `Deref` to `f32`** — intentional. Calling `.sin()` on a dereferenced
    /// `f32` would silently use the non-deterministic stdlib version, defeating the
    /// purpose of this type. Use `.0` to extract the raw value when needed.
    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    #[repr(transparent)]
    pub struct DeterministicF32(pub f32);

    impl From<f32> for DeterministicF32 {
        fn from(v: f32) -> Self {
            Self(v)
        }
    }

    impl From<DeterministicF32> for f32 {
        fn from(v: DeterministicF32) -> f32 {
            v.0
        }
    }

    impl From<DeterministicF32> for f64 {
        fn from(v: DeterministicF32) -> f64 {
            v.0.into()
        }
    }

    impl Add for DeterministicF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for DeterministicF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for DeterministicF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    impl Div for DeterministicF32 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(self.0 / rhs.0)
        }
    }

    impl Neg for DeterministicF32 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }

    impl Rem for DeterministicF32 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            Self(self.0 % rhs.0)
        }
    }

    impl DeterministicF32 {
        /// Return the square root using `libm::sqrtf`.
        ///
        /// Precision: delegates to `libm::sqrtf`. Results are deterministic and
        /// reproducible across platforms. `libm` provides a portable
        /// approximation; it does not guarantee correct rounding for all inputs.
        /// If you need strict ULP error bounds, use a library that documents
        /// them or perform numeric validation for your use-case.
        pub fn sqrt(self) -> Self {
            Self(libm::sqrtf(self.0))
        }

        /// Return the sine (radians) using `libm::sinf`.
        ///
        /// Precision: maps to `libm::sinf`. See the note on `sqrt` for general
        /// precision guidance; `libm`'s accuracy is implementation-specific.
        pub fn sin(self) -> Self {
            Self(libm::sinf(self.0))
        }

        /// Return the cosine (radians) using `libm::cosf`.
        ///
        /// Precision: maps to `libm::cosf`. See the `sqrt` note for general
        /// guidance on `libm`'s deterministic but approximate behaviour.
        pub fn cos(self) -> Self {
            Self(libm::cosf(self.0))
        }

        /// Return the tangent (radians) using `libm::tanf`.
        ///
        /// Precision: maps to `libm::tanf`. See the `sqrt` note for general
        /// guidance on `libm` accuracy and deterministic results.
        pub fn tan(self) -> Self {
            Self(libm::tanf(self.0))
        }

        /// Return arcsine (radians) using `libm::asinf`.
        ///
        /// Precision: maps to `libm::asinf`. See the `sqrt` note for the general
        /// precision caveat about `libm` implementations.
        pub fn asin(self) -> Self {
            Self(libm::asinf(self.0))
        }

        /// Return arccosine (radians) using `libm::acosf`.
        ///
        /// Precision: maps to `libm::acosf`. See `sqrt` for the general note.
        pub fn acos(self) -> Self {
            Self(libm::acosf(self.0))
        }

        /// Return arctangent (radians) using `libm::atanf`.
        ///
        /// Precision: maps to `libm::atanf`. See `sqrt` for general guidance.
        pub fn atan(self) -> Self {
            Self(libm::atanf(self.0))
        }

        /// Return `atan2(self, other)` using `libm::atan2f`.
        ///
        /// Precision: maps to `libm::atan2f`. See `sqrt` for the general note on
        /// `libm`'s deterministic but not strictly correctly-rounded behaviour.
        pub fn atan2(self, other: Self) -> Self {
            Self(libm::atan2f(self.0, other.0))
        }

        /// Return the exponential using `libm::expf`.
        ///
        /// Precision: maps to `libm::expf`. See `sqrt` for the general note on
        /// deterministic, implementation-specific accuracy.
        pub fn exp(self) -> Self {
            Self(libm::expf(self.0))
        }

        /// Return 2^x using `libm::exp2f`.
        ///
        /// Precision: maps to `libm::exp2f`. See `sqrt` for general caveats.
        pub fn exp2(self) -> Self {
            Self(libm::exp2f(self.0))
        }

        /// Return the natural logarithm using `libm::logf`.
        ///
        /// Precision: maps to `libm::logf`. See `sqrt` for general guidance.
        pub fn ln(self) -> Self {
            Self(libm::logf(self.0))
        }

        /// Return the base-2 logarithm using `libm::log2f`.
        ///
        /// Precision: maps to `libm::log2f`. See `sqrt` for the general note.
        pub fn log2(self) -> Self {
            Self(libm::log2f(self.0))
        }

        /// Return the base-10 logarithm using `libm::log10f`.
        ///
        /// Precision: maps to `libm::log10f`. See `sqrt` for general guidance.
        pub fn log10(self) -> Self {
            Self(libm::log10f(self.0))
        }

        /// Return `self.powf(exp)` using `libm::powf`.
        ///
        /// Precision: maps to `libm::powf`. See `sqrt` for the general note on
        /// `libm` accuracy and determinism.
        pub fn powf(self, exp: Self) -> Self {
            Self(libm::powf(self.0, exp.0))
        }

        /// Return the floor using `libm::floorf`.
        ///
        /// Precision: maps to `libm::floorf` (exact for IEEE floats where
        /// floor is defined). See `sqrt` for general guidance on other funcs.
        pub fn floor(self) -> Self {
            Self(libm::floorf(self.0))
        }

        /// Return the ceiling using `libm::ceilf`.
        ///
        /// Precision: maps to `libm::ceilf` (exact for IEEE floats where ceil
        /// is defined). See `sqrt` for general guidance on other functions.
        pub fn ceil(self) -> Self {
            Self(libm::ceilf(self.0))
        }

        /// Return the nearest integer using `libm::roundf`.
        ///
        /// Precision: maps to `libm::roundf`. See `sqrt` for the general note on
        /// `libm`'s deterministic approximations.
        pub fn round(self) -> Self {
            Self(libm::roundf(self.0))
        }

        /// Return the absolute value using `libm::fabsf`.
        ///
        /// Precision: maps to `libm::fabsf` (exact for IEEE floats). See `sqrt`
        /// for the general precision caveat about other transcendental funcs.
        pub fn abs(self) -> Self {
            Self(libm::fabsf(self.0))
        }

        pub fn signum(self) -> Self {
            if self.0.is_nan() {
                Self(self.0)
            } else {
                Self(libm::copysignf(1.0, self.0))
            }
        }

        pub fn is_sign_positive(self) -> bool {
            self.0.is_sign_positive()
        }

        pub fn is_sign_negative(self) -> bool {
            self.0.is_sign_negative()
        }

        pub fn trunc(self) -> Self {
            Self(libm::truncf(self.0))
        }

        pub fn fract(self) -> Self {
            Self(self.0 - libm::truncf(self.0))
        }

        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self(libm::fmaf(self.0, a.0, b.0))
        }

        pub fn recip(self) -> Self {
            Self(1.0 / self.0)
        }

        pub fn powi(self, n: i32) -> Self {
            Self(libm::powf(self.0, n as f32))
        }

        pub fn log(self, base: Self) -> Self {
            Self(libm::logf(self.0) / libm::logf(base.0))
        }

        pub fn to_degrees(self) -> Self {
            Self(self.0.to_degrees())
        }

        pub fn to_radians(self) -> Self {
            Self(self.0.to_radians())
        }

        pub fn max(self, other: Self) -> Self {
            if self.0.is_nan() {
                other
            } else if other.0.is_nan() {
                self
            } else if self.0 < other.0 {
                other
            } else {
                self
            }
        }

        pub fn min(self, other: Self) -> Self {
            if self.0.is_nan() {
                other
            } else if other.0.is_nan() {
                self
            } else if self.0 > other.0 {
                other
            } else {
                self
            }
        }

        pub fn abs_sub(self, other: Self) -> Self {
            Self(libm::fdimf(self.0, other.0))
        }

        pub fn cbrt(self) -> Self {
            Self(libm::cbrtf(self.0))
        }

        pub fn hypot(self, other: Self) -> Self {
            Self(libm::hypotf(self.0, other.0))
        }

        pub fn sin_cos(self) -> (Self, Self) {
            let (s, c) = libm::sincosf(self.0);
            (Self(s), Self(c))
        }

        pub fn exp_m1(self) -> Self {
            Self(libm::expm1f(self.0))
        }

        pub fn ln_1p(self) -> Self {
            Self(libm::log1pf(self.0))
        }

        pub fn sinh(self) -> Self {
            Self(libm::sinhf(self.0))
        }

        pub fn cosh(self) -> Self {
            Self(libm::coshf(self.0))
        }

        pub fn tanh(self) -> Self {
            Self(libm::tanhf(self.0))
        }

        pub fn asinh(self) -> Self {
            Self(libm::asinhf(self.0))
        }

        pub fn acosh(self) -> Self {
            Self(libm::acoshf(self.0))
        }

        pub fn atanh(self) -> Self {
            Self(libm::atanhf(self.0))
        }

        pub fn copysign(self, sign: Self) -> Self {
            Self(libm::copysignf(self.0, sign.0))
        }
    }

    // Num Traits:
    impl Zero for DeterministicF32 {
        fn zero() -> Self {
            Self(0.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl One for DeterministicF32 {
        fn one() -> Self {
            Self(1.0)
        }

        fn is_one(&self) -> bool {
            self.0 == 1.0
        }
    }

    impl ToPrimitive for DeterministicF32 {
        fn to_i64(&self) -> Option<i64> {
            Some(self.0 as i64)
        }

        fn to_u64(&self) -> Option<u64> {
            Some(self.0 as u64)
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0 as f64)
        }

        fn to_f32(&self) -> Option<f32> {
            Some(self.0)
        }
    }

    impl FromPrimitive for DeterministicF32 {
        fn from_i64(n: i64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_u64(n: u64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_f64(n: f64) -> Option<Self> {
            Some(Self(n as f32))
        }

        fn from_f32(n: f32) -> Option<Self> {
            Some(Self(n))
        }
    }

    impl NumCast for DeterministicF32 {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f32().map(Self)
        }
    }

    impl Num for DeterministicF32 {
        type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
        fn from_str_radix(
            src: &str,
            radix: u32,
        ) -> Result<Self, Self::FromStrRadixErr> {
            <f32 as Num>::from_str_radix(src, radix).map(Self)
        }
    }

    impl Signed for DeterministicF32 {
        fn abs(&self) -> Self {
            (*self).abs()
        }

        fn abs_sub(&self, other: &Self) -> Self {
            (*self).abs_sub(*other)
        }

        fn signum(&self) -> Self {
            (*self).signum()
        }

        fn is_positive(&self) -> bool {
            self.is_sign_positive()
        }

        fn is_negative(&self) -> bool {
            self.is_sign_negative()
        }
    }

    impl Float for DeterministicF32 {
        fn nan() -> Self {
            Self(f32::NAN)
        }
        fn infinity() -> Self {
            Self(f32::INFINITY)
        }
        fn neg_infinity() -> Self {
            Self(f32::NEG_INFINITY)
        }
        fn neg_zero() -> Self {
            Self(-0.0)
        }
        fn min_value() -> Self {
            Self(f32::MIN)
        }
        fn min_positive_value() -> Self {
            Self(f32::MIN_POSITIVE)
        }
        fn epsilon() -> Self {
            Self(f32::EPSILON)
        }
        fn max_value() -> Self {
            Self(f32::MAX)
        }

        fn is_nan(self) -> bool {
            self.0.is_nan()
        }
        fn is_infinite(self) -> bool {
            self.0.is_infinite()
        }
        fn is_finite(self) -> bool {
            self.0.is_finite()
        }
        fn is_normal(self) -> bool {
            self.0.is_normal()
        }
        fn classify(self) -> FpCategory {
            self.0.classify()
        }

        fn floor(self) -> Self {
            self.floor()
        }
        fn ceil(self) -> Self {
            self.ceil()
        }
        fn round(self) -> Self {
            self.round()
        }
        fn trunc(self) -> Self {
            self.trunc()
        }
        fn fract(self) -> Self {
            self.fract()
        }
        fn abs(self) -> Self {
            self.abs()
        }
        fn signum(self) -> Self {
            self.signum()
        }
        fn is_sign_positive(self) -> bool {
            self.is_sign_positive()
        }
        fn is_sign_negative(self) -> bool {
            self.is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            self.mul_add(a, b)
        }
        fn recip(self) -> Self {
            self.recip()
        }
        fn powi(self, n: i32) -> Self {
            self.powi(n)
        }
        fn powf(self, n: Self) -> Self {
            self.powf(n)
        }
        fn sqrt(self) -> Self {
            self.sqrt()
        }
        fn exp(self) -> Self {
            self.exp()
        }
        fn exp2(self) -> Self {
            self.exp2()
        }
        fn ln(self) -> Self {
            self.ln()
        }
        fn log(self, base: Self) -> Self {
            self.log(base)
        }
        fn log2(self) -> Self {
            self.log2()
        }
        fn log10(self) -> Self {
            self.log10()
        }
        fn to_degrees(self) -> Self {
            self.to_degrees()
        }
        fn to_radians(self) -> Self {
            self.to_radians()
        }
        fn max(self, other: Self) -> Self {
            self.max(other)
        }
        fn min(self, other: Self) -> Self {
            self.min(other)
        }
        fn abs_sub(self, other: Self) -> Self {
            self.abs_sub(other)
        }
        fn cbrt(self) -> Self {
            self.cbrt()
        }
        fn hypot(self, other: Self) -> Self {
            self.hypot(other)
        }

        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
        fn tan(self) -> Self {
            self.tan()
        }
        fn asin(self) -> Self {
            self.asin()
        }
        fn acos(self) -> Self {
            self.acos()
        }
        fn atan(self) -> Self {
            self.atan()
        }
        fn atan2(self, other: Self) -> Self {
            self.atan2(other)
        }
        fn sin_cos(self) -> (Self, Self) {
            self.sin_cos()
        }

        fn exp_m1(self) -> Self {
            self.exp_m1()
        }
        fn ln_1p(self) -> Self {
            self.ln_1p()
        }
        fn sinh(self) -> Self {
            self.sinh()
        }
        fn cosh(self) -> Self {
            self.cosh()
        }
        fn tanh(self) -> Self {
            self.tanh()
        }
        fn asinh(self) -> Self {
            self.asinh()
        }
        fn acosh(self) -> Self {
            self.acosh()
        }
        fn atanh(self) -> Self {
            self.atanh()
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            let bits: u32 = self.0.to_bits();
            let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
            let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
            let mantissa = if exponent == 0 {
                (bits & 0x7fffff) << 1
            } else {
                (bits & 0x7fffff) | 0x800000
            };
            exponent -= 127 + 23;
            (mantissa as u64, exponent, sign)
        }

        fn copysign(self, sign: Self) -> Self {
            self.copysign(sign)
        }
    }
}
mod deterministic_f64 {
    use super::deterministic_f32::DeterministicF32;
    use core::num::FpCategory;
    use core::ops::Rem;
    use num_traits::{
        Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
    };
    use std::ops::{Add, Div, Mul, Neg, Sub};

    /// A wrapper around `f64` that routes all transcendental operations through
    /// [`libm`], producing bit-identical results on every platform.
    ///
    /// Standard-library transcendentals (`f64::sin`, etc.) may use hardware
    /// instructions whose precision varies by CPU and compiler. This type delegates
    /// to `libm`'s portable pure-Rust implementations instead, so results are
    /// reproducible across machines — useful for deterministic simulation,
    /// serialised state, or cross-platform tests. There are errors in here. If you
    /// want a specific level of precision, it would be better to use a library that
    /// either specifically documents their ULP or
    ///
    /// Arithmetic operators (`+`, `-`, `*`, `/`) use ordinary IEEE 754 operations,
    /// which are already correctly-rounded and cross-platform identical.
    ///
    /// **No `Deref` to `f64`** — intentional. Calling `.sin()` on a dereferenced
    /// `f64` would silently use the non-deterministic stdlib version, defeating the
    /// purpose of this type. Use `.0` to extract the raw value when needed.
    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    #[repr(transparent)]
    pub struct DeterministicF64(pub f64);

    impl From<f64> for DeterministicF64 {
        fn from(v: f64) -> Self {
            Self(v)
        }
    }

    impl From<DeterministicF64> for f64 {
        fn from(v: DeterministicF64) -> f64 {
            v.0
        }
    }

    impl From<f32> for DeterministicF64 {
        fn from(value: f32) -> Self {
            Self(value.into())
        }
    }

    impl From<DeterministicF32> for DeterministicF64 {
        fn from(value: DeterministicF32) -> Self {
            Self(value.0.into())
        }
    }

    impl Add for DeterministicF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for DeterministicF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for DeterministicF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    impl Div for DeterministicF64 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(self.0 / rhs.0)
        }
    }

    impl Neg for DeterministicF64 {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }

    impl Rem for DeterministicF64 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            Self(self.0 % rhs.0)
        }
    }

    impl DeterministicF64 {
        /// Return the square root using `libm::sqrt`.
        ///
        /// Precision: delegates to `libm::sqrt`. Results are deterministic and
        /// reproducible across platforms. `libm` implements portable
        /// approximations and does not guarantee correct rounding for all
        /// inputs. If strict ULP bounds are required, choose a library that
        /// documents them or perform numeric validation.
        pub fn sqrt(self) -> Self {
            Self(libm::sqrt(self.0))
        }

        /// Return the absolute value using `libm::fabs`.
        ///
        /// Precision: maps to `libm::fabs` (exact for IEEE floats). See `sqrt`
        /// for the general precision caveat about other transcendental funcs.
        pub fn abs(self) -> Self {
            Self(libm::fabs(self.0))
        }

        pub fn signum(self) -> Self {
            if self.0.is_nan() {
                Self(self.0)
            } else {
                Self(libm::copysign(1.0, self.0))
            }
        }

        pub fn is_sign_positive(self) -> bool {
            self.0.is_sign_positive()
        }

        pub fn is_sign_negative(self) -> bool {
            self.0.is_sign_negative()
        }

        pub fn trunc(self) -> Self {
            Self(libm::trunc(self.0))
        }

        pub fn fract(self) -> Self {
            Self(self.0 - libm::trunc(self.0))
        }

        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self(libm::fma(self.0, a.0, b.0))
        }

        pub fn recip(self) -> Self {
            Self(1.0 / self.0)
        }

        pub fn powi(self, n: i32) -> Self {
            Self(libm::pow(self.0, n as f64))
        }

        pub fn log(self, base: Self) -> Self {
            Self(libm::log(self.0) / libm::log(base.0))
        }

        pub fn to_degrees(self) -> Self {
            Self(self.0.to_degrees())
        }

        pub fn to_radians(self) -> Self {
            Self(self.0.to_radians())
        }

        pub fn max(self, other: Self) -> Self {
            if self.0.is_nan() {
                other
            } else if other.0.is_nan() {
                self
            } else if self.0 < other.0 {
                other
            } else {
                self
            }
        }

        pub fn min(self, other: Self) -> Self {
            if self.0.is_nan() {
                other
            } else if other.0.is_nan() {
                self
            } else if self.0 > other.0 {
                other
            } else {
                self
            }
        }

        pub fn abs_sub(self, other: Self) -> Self {
            Self(libm::fdim(self.0, other.0))
        }

        pub fn cbrt(self) -> Self {
            Self(libm::cbrt(self.0))
        }

        pub fn hypot(self, other: Self) -> Self {
            Self(libm::hypot(self.0, other.0))
        }

        pub fn sin_cos(self) -> (Self, Self) {
            let (s, c) = libm::sincos(self.0);
            (Self(s), Self(c))
        }

        pub fn exp_m1(self) -> Self {
            Self(libm::expm1(self.0))
        }

        pub fn ln_1p(self) -> Self {
            Self(libm::log1p(self.0))
        }

        pub fn sinh(self) -> Self {
            Self(libm::sinh(self.0))
        }

        pub fn cosh(self) -> Self {
            Self(libm::cosh(self.0))
        }

        pub fn tanh(self) -> Self {
            Self(libm::tanh(self.0))
        }

        pub fn asinh(self) -> Self {
            Self(libm::asinh(self.0))
        }

        pub fn acosh(self) -> Self {
            Self(libm::acosh(self.0))
        }

        pub fn atanh(self) -> Self {
            Self(libm::atanh(self.0))
        }

        pub fn copysign(self, sign: Self) -> Self {
            Self(libm::copysign(self.0, sign.0))
        }

        /// Return the sine (radians) using `libm::sin`.
        ///
        /// Precision: maps to `libm::sin`. See the `sqrt` note for general
        /// guidance on `libm` accuracy and determinism.
        pub fn sin(self) -> Self {
            Self(libm::sin(self.0))
        }

        /// Return the cosine (radians) using `libm::cos`.
        ///
        /// Precision: maps to `libm::cos`. See `sqrt` for the general note on
        /// `libm`'s approximations.
        pub fn cos(self) -> Self {
            Self(libm::cos(self.0))
        }

        /// Return the tangent (radians) using `libm::tan`.
        ///
        /// Precision: maps to `libm::tan`. See `sqrt` for general guidance.
        pub fn tan(self) -> Self {
            Self(libm::tan(self.0))
        }

        /// Return arcsine (radians) using `libm::asin`.
        ///
        /// Precision: maps to `libm::asin`. See the `sqrt` note for the general
        /// caveat about `libm` accuracy.
        pub fn asin(self) -> Self {
            Self(libm::asin(self.0))
        }

        /// Return arccosine (radians) using `libm::acos`.
        ///
        /// Precision: maps to `libm::acos`. See `sqrt` for the general note.
        pub fn acos(self) -> Self {
            Self(libm::acos(self.0))
        }

        /// Return arctangent (radians) using `libm::atan`.
        ///
        /// Precision: maps to `libm::atan`. See `sqrt` for general guidance.
        pub fn atan(self) -> Self {
            Self(libm::atan(self.0))
        }

        /// Return `atan2(self, other)` using `libm::atan2`.
        ///
        /// Precision: maps to `libm::atan2`. See `sqrt` for the general note on
        /// `libm`'s deterministic approximations.
        pub fn atan2(self, other: Self) -> Self {
            Self(libm::atan2(self.0, other.0))
        }

        /// Return the exponential using `libm::exp`.
        ///
        /// Precision: maps to `libm::exp`. See `sqrt` for the general note on
        /// implementation-specific accuracy.
        pub fn exp(self) -> Self {
            Self(libm::exp(self.0))
        }

        /// Return 2^x using `libm::exp2`.
        ///
        /// Precision: maps to `libm::exp2`. See `sqrt` for general caveats.
        pub fn exp2(self) -> Self {
            Self(libm::exp2(self.0))
        }

        /// Return the natural logarithm using `libm::log`.
        ///
        /// Precision: maps to `libm::log`. See `sqrt` for general guidance.
        pub fn ln(self) -> Self {
            Self(libm::log(self.0))
        }

        /// Return the base-2 logarithm using `libm::log2`.
        ///
        /// Precision: maps to `libm::log2`. See `sqrt` for the general note.
        pub fn log2(self) -> Self {
            Self(libm::log2(self.0))
        }

        /// Return the base-10 logarithm using `libm::log10`.
        ///
        /// Precision: maps to `libm::log10`. See `sqrt` for general guidance.
        pub fn log10(self) -> Self {
            Self(libm::log10(self.0))
        }

        /// Return `self.powf(exp)` using `libm::pow`.
        ///
        /// Precision: maps to `libm::pow`. See `sqrt` for the general note on
        /// `libm` accuracy and determinism.
        pub fn powf(self, exp: Self) -> Self {
            Self(libm::pow(self.0, exp.0))
        }

        /// Return the floor using `libm::floor`.
        ///
        /// Precision: maps to `libm::floor` (exact for IEEE floats where floor
        /// is defined). See `sqrt` for general guidance on other functions.
        pub fn floor(self) -> Self {
            Self(libm::floor(self.0))
        }

        /// Return the ceiling using `libm::ceil`.
        ///
        /// Precision: maps to `libm::ceil` (exact for IEEE floats where ceil is
        /// defined). See `sqrt` for general guidance on other functions.
        pub fn ceil(self) -> Self {
            Self(libm::ceil(self.0))
        }

        /// Return the nearest integer using `libm::round`.
        ///
        /// Precision: maps to `libm::round`. See `sqrt` for the general note on
        /// `libm`'s deterministic approximations.
        pub fn round(self) -> Self {
            Self(libm::round(self.0))
        }
    }

    // Num traits:

    impl Zero for DeterministicF64 {
        fn zero() -> Self {
            Self(0.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl One for DeterministicF64 {
        fn one() -> Self {
            Self(1.0)
        }

        fn is_one(&self) -> bool {
            self.0 == 1.0
        }
    }

    impl ToPrimitive for DeterministicF64 {
        fn to_i64(&self) -> Option<i64> {
            Some(self.0 as i64)
        }

        fn to_u64(&self) -> Option<u64> {
            Some(self.0 as u64)
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0)
        }

        fn to_f32(&self) -> Option<f32> {
            Some(self.0 as f32)
        }
    }

    impl FromPrimitive for DeterministicF64 {
        fn from_i64(n: i64) -> Option<Self> {
            Some(Self(n as f64))
        }

        fn from_u64(n: u64) -> Option<Self> {
            Some(Self(n as f64))
        }

        fn from_f64(n: f64) -> Option<Self> {
            Some(Self(n))
        }

        fn from_f32(n: f32) -> Option<Self> {
            Some(Self(n.into()))
        }
    }

    impl NumCast for DeterministicF64 {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f64().map(Self)
        }
    }

    impl Num for DeterministicF64 {
        type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
        fn from_str_radix(
            src: &str,
            radix: u32,
        ) -> Result<Self, Self::FromStrRadixErr> {
            <f64 as Num>::from_str_radix(src, radix).map(Self)
        }
    }

    impl Signed for DeterministicF64 {
        fn abs(&self) -> Self {
            (*self).abs()
        }

        fn abs_sub(&self, other: &Self) -> Self {
            (*self).abs_sub(*other)
        }

        fn signum(&self) -> Self {
            (*self).signum()
        }

        fn is_positive(&self) -> bool {
            self.is_sign_positive()
        }

        fn is_negative(&self) -> bool {
            self.is_sign_negative()
        }
    }

    impl Float for DeterministicF64 {
        fn nan() -> Self {
            Self(f64::NAN)
        }
        fn infinity() -> Self {
            Self(f64::INFINITY)
        }
        fn neg_infinity() -> Self {
            Self(f64::NEG_INFINITY)
        }
        fn neg_zero() -> Self {
            Self(-0.0)
        }
        fn min_value() -> Self {
            Self(f64::MIN)
        }
        fn min_positive_value() -> Self {
            Self(f64::MIN_POSITIVE)
        }
        fn epsilon() -> Self {
            Self(f64::EPSILON)
        }
        fn max_value() -> Self {
            Self(f64::MAX)
        }

        fn is_nan(self) -> bool {
            self.0.is_nan()
        }
        fn is_infinite(self) -> bool {
            self.0.is_infinite()
        }
        fn is_finite(self) -> bool {
            self.0.is_finite()
        }
        fn is_normal(self) -> bool {
            self.0.is_normal()
        }
        fn classify(self) -> FpCategory {
            self.0.classify()
        }

        fn floor(self) -> Self {
            self.floor()
        }
        fn ceil(self) -> Self {
            self.ceil()
        }
        fn round(self) -> Self {
            self.round()
        }
        fn trunc(self) -> Self {
            self.trunc()
        }
        fn fract(self) -> Self {
            self.fract()
        }
        fn abs(self) -> Self {
            self.abs()
        }
        fn signum(self) -> Self {
            self.signum()
        }
        fn is_sign_positive(self) -> bool {
            self.is_sign_positive()
        }
        fn is_sign_negative(self) -> bool {
            self.is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            self.mul_add(a, b)
        }
        fn recip(self) -> Self {
            self.recip()
        }
        fn powi(self, n: i32) -> Self {
            self.powi(n)
        }
        fn powf(self, n: Self) -> Self {
            self.powf(n)
        }
        fn sqrt(self) -> Self {
            self.sqrt()
        }
        fn exp(self) -> Self {
            self.exp()
        }
        fn exp2(self) -> Self {
            self.exp2()
        }
        fn ln(self) -> Self {
            self.ln()
        }
        fn log(self, base: Self) -> Self {
            self.log(base)
        }
        fn log2(self) -> Self {
            self.log2()
        }
        fn log10(self) -> Self {
            self.log10()
        }
        fn to_degrees(self) -> Self {
            self.to_degrees()
        }
        fn to_radians(self) -> Self {
            self.to_radians()
        }
        fn max(self, other: Self) -> Self {
            self.max(other)
        }
        fn min(self, other: Self) -> Self {
            self.min(other)
        }
        fn abs_sub(self, other: Self) -> Self {
            self.abs_sub(other)
        }
        fn cbrt(self) -> Self {
            self.cbrt()
        }
        fn hypot(self, other: Self) -> Self {
            self.hypot(other)
        }

        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
        fn tan(self) -> Self {
            self.tan()
        }
        fn asin(self) -> Self {
            self.asin()
        }
        fn acos(self) -> Self {
            self.acos()
        }
        fn atan(self) -> Self {
            self.atan()
        }
        fn atan2(self, other: Self) -> Self {
            self.atan2(other)
        }
        fn sin_cos(self) -> (Self, Self) {
            self.sin_cos()
        }

        fn exp_m1(self) -> Self {
            self.exp_m1()
        }
        fn ln_1p(self) -> Self {
            self.ln_1p()
        }
        fn sinh(self) -> Self {
            self.sinh()
        }
        fn cosh(self) -> Self {
            self.cosh()
        }
        fn tanh(self) -> Self {
            self.tanh()
        }
        fn asinh(self) -> Self {
            self.asinh()
        }
        fn acosh(self) -> Self {
            self.acosh()
        }
        fn atanh(self) -> Self {
            self.atanh()
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            let bits: u64 = self.0.to_bits();
            let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
            let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
            let mantissa = if exponent == 0 {
                (bits & 0xfffffffffffff) << 1
            } else {
                (bits & 0xfffffffffffff) | 0x10000000000000
            };
            exponent -= 1023 + 52;
            (mantissa, exponent, sign)
        }

        fn copysign(self, sign: Self) -> Self {
            self.copysign(sign)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{DeterministicF32, DeterministicF64};
    use proptest::prelude::*;

    const F32_SPECIAL_BITS: &[u32] = &[
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x4049_0fdb,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7fa0_0001,
        0xffc0_0000,
    ];

    const F64_SPECIAL_BITS: &[u64] = &[
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x4009_21fb_5444_2d18,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0x7ff4_0000_0000_0001,
        0xfff8_0000_0000_0000,
    ];

    fn next_u32(state: &mut u32) -> u32 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        *state = value;
        value
    }

    fn next_u64(state: &mut u64) -> u64 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        value
    }

    fn assert_f32_bits_eq(actual: f32, expected: f32, label: &str) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}: actual={actual:?} expected={expected:?} \
             actual_bits={:#010x} expected_bits={:#010x}",
            actual.to_bits(),
            expected.to_bits(),
        );
    }

    fn assert_f64_bits_eq(actual: f64, expected: f64, label: &str) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}: actual={actual:?} expected={expected:?} \
             actual_bits={:#018x} expected_bits={:#018x}",
            actual.to_bits(),
            expected.to_bits(),
        );
    }

    fn for_each_f32_sample(mut test: impl FnMut(f32)) {
        for &bits in F32_SPECIAL_BITS {
            test(f32::from_bits(bits));
        }

        let mut state = 0x243f_6a88;
        for _ in 0..8_192 {
            test(f32::from_bits(next_u32(&mut state)));
        }
    }

    fn for_each_f64_sample(mut test: impl FnMut(f64)) {
        for &bits in F64_SPECIAL_BITS {
            test(f64::from_bits(bits));
        }

        let mut state = 0x243f_6a88_85a3_08d3;
        for _ in 0..8_192 {
            test(f64::from_bits(next_u64(&mut state)));
        }
    }

    fn for_each_f32_pair(mut test: impl FnMut(f32, f32)) {
        for &left in F32_SPECIAL_BITS {
            for &right in F32_SPECIAL_BITS {
                test(f32::from_bits(left), f32::from_bits(right));
            }
        }

        let mut left_state = 0x1319_8a2e;
        let mut right_state = 0x0370_7344;
        for _ in 0..4_096 {
            let left = f32::from_bits(next_u32(&mut left_state));
            let right = f32::from_bits(next_u32(&mut right_state));
            test(left, right);
        }
    }

    fn for_each_f64_pair(mut test: impl FnMut(f64, f64)) {
        for &left in F64_SPECIAL_BITS {
            for &right in F64_SPECIAL_BITS {
                test(f64::from_bits(left), f64::from_bits(right));
            }
        }

        let mut left_state = 0x1319_8a2e_0370_7344;
        let mut right_state = 0xa409_3822_299f_31d0;
        for _ in 0..4_096 {
            let left = f64::from_bits(next_u64(&mut left_state));
            let right = f64::from_bits(next_u64(&mut right_state));
            test(left, right);
        }
    }

    #[test]
    fn deterministic_f32_arithmetic_matches_ieee_bits() {
        for_each_f32_sample(|value| {
            assert_f32_bits_eq((-DeterministicF32(value)).0, -value, "f32 neg");
        });

        for_each_f32_pair(|left, right| {
            assert_f32_bits_eq(
                (DeterministicF32(left) + DeterministicF32(right)).0,
                left + right,
                "f32 add",
            );
            assert_f32_bits_eq(
                (DeterministicF32(left) - DeterministicF32(right)).0,
                left - right,
                "f32 sub",
            );
            assert_f32_bits_eq(
                (DeterministicF32(left) * DeterministicF32(right)).0,
                left * right,
                "f32 mul",
            );
            assert_f32_bits_eq(
                (DeterministicF32(left) / DeterministicF32(right)).0,
                left / right,
                "f32 div",
            );
        });
    }

    #[test]
    fn deterministic_f32_extended_unary_methods_match_libm_bits() {
        for_each_f32_sample(|value| {
            let w = DeterministicF32(value);
            assert_f32_bits_eq(
                w.trunc().0,
                libm::truncf(value),
                "f32 trunc",
            );
            assert_f32_bits_eq(
                w.fract().0,
                value - libm::truncf(value),
                "f32 fract",
            );
            assert_f32_bits_eq(
                w.cbrt().0,
                libm::cbrtf(value),
                "f32 cbrt",
            );
            assert_f32_bits_eq(
                w.exp_m1().0,
                libm::expm1f(value),
                "f32 exp_m1",
            );
            assert_f32_bits_eq(
                w.ln_1p().0,
                libm::log1pf(value),
                "f32 ln_1p",
            );
            assert_f32_bits_eq(
                w.sinh().0,
                libm::sinhf(value),
                "f32 sinh",
            );
            assert_f32_bits_eq(
                w.cosh().0,
                libm::coshf(value),
                "f32 cosh",
            );
            assert_f32_bits_eq(
                w.tanh().0,
                libm::tanhf(value),
                "f32 tanh",
            );
            assert_f32_bits_eq(
                w.asinh().0,
                libm::asinhf(value),
                "f32 asinh",
            );
            assert_f32_bits_eq(
                w.acosh().0,
                libm::acoshf(value),
                "f32 acosh",
            );
            assert_f32_bits_eq(
                w.atanh().0,
                libm::atanhf(value),
                "f32 atanh",
            );
            let (ws, wc) = w.sin_cos();
            let (ls, lc) = libm::sincosf(value);
            assert_f32_bits_eq(ws.0, ls, "f32 sin_cos sin");
            assert_f32_bits_eq(wc.0, lc, "f32 sin_cos cos");
        });
    }

    #[test]
    fn deterministic_f32_extended_binary_methods_match_libm_bits() {
        for_each_f32_pair(|left, right| {
            let wl = DeterministicF32(left);
            let wr = DeterministicF32(right);
            assert_f32_bits_eq(
                wl.hypot(wr).0,
                libm::hypotf(left, right),
                "f32 hypot",
            );
            assert_f32_bits_eq(
                wl.abs_sub(wr).0,
                libm::fdimf(left, right),
                "f32 abs_sub",
            );
            assert_f32_bits_eq(
                wl.copysign(wr).0,
                libm::copysignf(left, right),
                "f32 copysign",
            );
            assert_f32_bits_eq(
                wl.log(wr).0,
                libm::logf(left) / libm::logf(right),
                "f32 log(base)",
            );
        });

        for_each_f32_sample(|value| {
            let w = DeterministicF32(value);
            assert_f32_bits_eq(
                w.recip().0,
                1.0 / value,
                "f32 recip",
            );
            assert_f32_bits_eq(
                w.to_degrees().0,
                value.to_degrees(),
                "f32 to_degrees",
            );
            assert_f32_bits_eq(
                w.to_radians().0,
                value.to_radians(),
                "f32 to_radians",
            );
        });
    }

    #[test]
    fn deterministic_f64_extended_unary_methods_match_libm_bits() {
        for_each_f64_sample(|value| {
            let w = DeterministicF64(value);
            assert_f64_bits_eq(
                w.trunc().0,
                libm::trunc(value),
                "f64 trunc",
            );
            assert_f64_bits_eq(
                w.fract().0,
                value - libm::trunc(value),
                "f64 fract",
            );
            assert_f64_bits_eq(
                w.cbrt().0,
                libm::cbrt(value),
                "f64 cbrt",
            );
            assert_f64_bits_eq(
                w.exp_m1().0,
                libm::expm1(value),
                "f64 exp_m1",
            );
            assert_f64_bits_eq(
                w.ln_1p().0,
                libm::log1p(value),
                "f64 ln_1p",
            );
            assert_f64_bits_eq(
                w.sinh().0,
                libm::sinh(value),
                "f64 sinh",
            );
            assert_f64_bits_eq(
                w.cosh().0,
                libm::cosh(value),
                "f64 cosh",
            );
            assert_f64_bits_eq(
                w.tanh().0,
                libm::tanh(value),
                "f64 tanh",
            );
            assert_f64_bits_eq(
                w.asinh().0,
                libm::asinh(value),
                "f64 asinh",
            );
            assert_f64_bits_eq(
                w.acosh().0,
                libm::acosh(value),
                "f64 acosh",
            );
            assert_f64_bits_eq(
                w.atanh().0,
                libm::atanh(value),
                "f64 atanh",
            );
            let (ws, wc) = w.sin_cos();
            let (ls, lc) = libm::sincos(value);
            assert_f64_bits_eq(ws.0, ls, "f64 sin_cos sin");
            assert_f64_bits_eq(wc.0, lc, "f64 sin_cos cos");
        });
    }

    #[test]
    fn deterministic_f64_extended_binary_methods_match_libm_bits() {
        for_each_f64_pair(|left, right| {
            let wl = DeterministicF64(left);
            let wr = DeterministicF64(right);
            assert_f64_bits_eq(
                wl.hypot(wr).0,
                libm::hypot(left, right),
                "f64 hypot",
            );
            assert_f64_bits_eq(
                wl.abs_sub(wr).0,
                libm::fdim(left, right),
                "f64 abs_sub",
            );
            assert_f64_bits_eq(
                wl.copysign(wr).0,
                libm::copysign(left, right),
                "f64 copysign",
            );
            assert_f64_bits_eq(
                wl.log(wr).0,
                libm::log(left) / libm::log(right),
                "f64 log(base)",
            );
        });

        for_each_f64_sample(|value| {
            let w = DeterministicF64(value);
            assert_f64_bits_eq(
                w.recip().0,
                1.0 / value,
                "f64 recip",
            );
            assert_f64_bits_eq(
                w.to_degrees().0,
                value.to_degrees(),
                "f64 to_degrees",
            );
            assert_f64_bits_eq(
                w.to_radians().0,
                value.to_radians(),
                "f64 to_radians",
            );
        });
    }

    #[test]
    fn deterministic_f32_unary_methods_match_libm_bits() {
        for_each_f32_sample(|value| {
            let wrapped = DeterministicF32(value);

            assert_f32_bits_eq(
                wrapped.sqrt().0,
                libm::sqrtf(value),
                "f32 sqrt",
            );
            assert_f32_bits_eq(wrapped.sin().0, libm::sinf(value), "f32 sin");
            assert_f32_bits_eq(wrapped.cos().0, libm::cosf(value), "f32 cos");
            assert_f32_bits_eq(wrapped.tan().0, libm::tanf(value), "f32 tan");
            assert_f32_bits_eq(
                wrapped.asin().0,
                libm::asinf(value),
                "f32 asin",
            );
            assert_f32_bits_eq(
                wrapped.acos().0,
                libm::acosf(value),
                "f32 acos",
            );
            assert_f32_bits_eq(
                wrapped.atan().0,
                libm::atanf(value),
                "f32 atan",
            );
            assert_f32_bits_eq(wrapped.exp().0, libm::expf(value), "f32 exp");
            assert_f32_bits_eq(
                wrapped.exp2().0,
                libm::exp2f(value),
                "f32 exp2",
            );
            assert_f32_bits_eq(wrapped.ln().0, libm::logf(value), "f32 ln");
            assert_f32_bits_eq(
                wrapped.log2().0,
                libm::log2f(value),
                "f32 log2",
            );
            assert_f32_bits_eq(
                wrapped.log10().0,
                libm::log10f(value),
                "f32 log10",
            );
            assert_f32_bits_eq(
                wrapped.floor().0,
                libm::floorf(value),
                "f32 floor",
            );
            assert_f32_bits_eq(
                wrapped.ceil().0,
                libm::ceilf(value),
                "f32 ceil",
            );
            assert_f32_bits_eq(
                wrapped.round().0,
                libm::roundf(value),
                "f32 round",
            );
            assert_f32_bits_eq(wrapped.abs().0, libm::fabsf(value), "f32 abs");
        });
    }

    #[test]
    fn deterministic_f32_binary_methods_match_libm_bits() {
        for_each_f32_pair(|left, right| {
            let wrapped_left = DeterministicF32(left);
            let wrapped_right = DeterministicF32(right);

            assert_f32_bits_eq(
                wrapped_left.atan2(wrapped_right).0,
                libm::atan2f(left, right),
                "f32 atan2",
            );
            assert_f32_bits_eq(
                wrapped_left.powf(wrapped_right).0,
                libm::powf(left, right),
                "f32 powf",
            );
        });
    }

    #[test]
    fn deterministic_f64_arithmetic_matches_ieee_bits() {
        for_each_f64_sample(|value| {
            assert_f64_bits_eq((-DeterministicF64(value)).0, -value, "f64 neg");
        });

        for_each_f64_pair(|left, right| {
            assert_f64_bits_eq(
                (DeterministicF64(left) + DeterministicF64(right)).0,
                left + right,
                "f64 add",
            );
            assert_f64_bits_eq(
                (DeterministicF64(left) - DeterministicF64(right)).0,
                left - right,
                "f64 sub",
            );
            assert_f64_bits_eq(
                (DeterministicF64(left) * DeterministicF64(right)).0,
                left * right,
                "f64 mul",
            );
            assert_f64_bits_eq(
                (DeterministicF64(left) / DeterministicF64(right)).0,
                left / right,
                "f64 div",
            );
        });
    }

    #[test]
    fn deterministic_f64_unary_methods_match_libm_bits() {
        for_each_f64_sample(|value| {
            let wrapped = DeterministicF64(value);

            assert_f64_bits_eq(wrapped.sqrt().0, libm::sqrt(value), "f64 sqrt");
            assert_f64_bits_eq(wrapped.sin().0, libm::sin(value), "f64 sin");
            assert_f64_bits_eq(wrapped.cos().0, libm::cos(value), "f64 cos");
            assert_f64_bits_eq(wrapped.tan().0, libm::tan(value), "f64 tan");
            assert_f64_bits_eq(wrapped.asin().0, libm::asin(value), "f64 asin");
            assert_f64_bits_eq(wrapped.acos().0, libm::acos(value), "f64 acos");
            assert_f64_bits_eq(wrapped.atan().0, libm::atan(value), "f64 atan");
            assert_f64_bits_eq(wrapped.exp().0, libm::exp(value), "f64 exp");
            assert_f64_bits_eq(wrapped.exp2().0, libm::exp2(value), "f64 exp2");
            assert_f64_bits_eq(wrapped.ln().0, libm::log(value), "f64 ln");
            assert_f64_bits_eq(wrapped.log2().0, libm::log2(value), "f64 log2");
            assert_f64_bits_eq(
                wrapped.log10().0,
                libm::log10(value),
                "f64 log10",
            );
            assert_f64_bits_eq(
                wrapped.floor().0,
                libm::floor(value),
                "f64 floor",
            );
            assert_f64_bits_eq(wrapped.ceil().0, libm::ceil(value), "f64 ceil");
            assert_f64_bits_eq(
                wrapped.round().0,
                libm::round(value),
                "f64 round",
            );
            assert_f64_bits_eq(wrapped.abs().0, libm::fabs(value), "f64 abs");
        });
    }

    #[test]
    fn deterministic_f64_binary_methods_match_libm_bits() {
        for_each_f64_pair(|left, right| {
            let wrapped_left = DeterministicF64(left);
            let wrapped_right = DeterministicF64(right);

            assert_f64_bits_eq(
                wrapped_left.atan2(wrapped_right).0,
                libm::atan2(left, right),
                "f64 atan2",
            );
            assert_f64_bits_eq(
                wrapped_left.powf(wrapped_right).0,
                libm::pow(left, right),
                "f64 powf",
            );
        });
    }

    proptest! {
        #[test]
        fn proptest_f32_unary_matches_libm(x in any::<f32>()) {
            let w = DeterministicF32(x);
            prop_assert_eq!(w.sqrt().0.to_bits(), libm::sqrtf(x).to_bits());
            prop_assert_eq!(w.sin().0.to_bits(), libm::sinf(x).to_bits());
            prop_assert_eq!(w.cos().0.to_bits(), libm::cosf(x).to_bits());
            prop_assert_eq!(w.tan().0.to_bits(), libm::tanf(x).to_bits());
            prop_assert_eq!(w.asin().0.to_bits(), libm::asinf(x).to_bits());
            prop_assert_eq!(w.acos().0.to_bits(), libm::acosf(x).to_bits());
            prop_assert_eq!(w.atan().0.to_bits(), libm::atanf(x).to_bits());
            prop_assert_eq!(w.exp().0.to_bits(), libm::expf(x).to_bits());
            prop_assert_eq!(w.exp2().0.to_bits(), libm::exp2f(x).to_bits());
            prop_assert_eq!(w.ln().0.to_bits(), libm::logf(x).to_bits());
            prop_assert_eq!(w.log2().0.to_bits(), libm::log2f(x).to_bits());
            prop_assert_eq!(w.log10().0.to_bits(), libm::log10f(x).to_bits());
            prop_assert_eq!(w.floor().0.to_bits(), libm::floorf(x).to_bits());
            prop_assert_eq!(w.ceil().0.to_bits(), libm::ceilf(x).to_bits());
            prop_assert_eq!(w.round().0.to_bits(), libm::roundf(x).to_bits());
            prop_assert_eq!(w.abs().0.to_bits(), libm::fabsf(x).to_bits());
        }

        #[test]
        fn proptest_f32_arithmetic_matches_ieee(a in any::<f32>(), b in any::<f32>()) {
            prop_assert_eq!((DeterministicF32(a) + DeterministicF32(b)).0.to_bits(), (a + b).to_bits());
            prop_assert_eq!((DeterministicF32(a) - DeterministicF32(b)).0.to_bits(), (a - b).to_bits());
            prop_assert_eq!((DeterministicF32(a) * DeterministicF32(b)).0.to_bits(), (a * b).to_bits());
            prop_assert_eq!((DeterministicF32(a) / DeterministicF32(b)).0.to_bits(), (a / b).to_bits());
        }

        #[test]
        fn proptest_f32_binary_matches_libm(a in any::<f32>(), b in any::<f32>()) {
            prop_assert_eq!(DeterministicF32(a).atan2(DeterministicF32(b)).0.to_bits(), libm::atan2f(a, b).to_bits());
            prop_assert_eq!(DeterministicF32(a).powf(DeterministicF32(b)).0.to_bits(), libm::powf(a, b).to_bits());
        }

        #[test]
        fn proptest_f64_unary_matches_libm(x in any::<f64>()) {
            let w = DeterministicF64(x);
            prop_assert_eq!(w.sqrt().0.to_bits(), libm::sqrt(x).to_bits());
            prop_assert_eq!(w.sin().0.to_bits(), libm::sin(x).to_bits());
            prop_assert_eq!(w.cos().0.to_bits(), libm::cos(x).to_bits());
            prop_assert_eq!(w.tan().0.to_bits(), libm::tan(x).to_bits());
            prop_assert_eq!(w.asin().0.to_bits(), libm::asin(x).to_bits());
            prop_assert_eq!(w.acos().0.to_bits(), libm::acos(x).to_bits());
            prop_assert_eq!(w.atan().0.to_bits(), libm::atan(x).to_bits());
            prop_assert_eq!(w.exp().0.to_bits(), libm::exp(x).to_bits());
            prop_assert_eq!(w.exp2().0.to_bits(), libm::exp2(x).to_bits());
            prop_assert_eq!(w.ln().0.to_bits(), libm::log(x).to_bits());
            prop_assert_eq!(w.log2().0.to_bits(), libm::log2(x).to_bits());
            prop_assert_eq!(w.log10().0.to_bits(), libm::log10(x).to_bits());
            prop_assert_eq!(w.floor().0.to_bits(), libm::floor(x).to_bits());
            prop_assert_eq!(w.ceil().0.to_bits(), libm::ceil(x).to_bits());
            prop_assert_eq!(w.round().0.to_bits(), libm::round(x).to_bits());
            prop_assert_eq!(w.abs().0.to_bits(), libm::fabs(x).to_bits());
        }

        #[test]
        fn proptest_f64_arithmetic_matches_ieee(a in any::<f64>(), b in any::<f64>()) {
            prop_assert_eq!((DeterministicF64(a) + DeterministicF64(b)).0.to_bits(), (a + b).to_bits());
            prop_assert_eq!((DeterministicF64(a) - DeterministicF64(b)).0.to_bits(), (a - b).to_bits());
            prop_assert_eq!((DeterministicF64(a) * DeterministicF64(b)).0.to_bits(), (a * b).to_bits());
            prop_assert_eq!((DeterministicF64(a) / DeterministicF64(b)).0.to_bits(), (a / b).to_bits());
        }

        #[test]
        fn proptest_f64_binary_matches_libm(a in any::<f64>(), b in any::<f64>()) {
            prop_assert_eq!(DeterministicF64(a).atan2(DeterministicF64(b)).0.to_bits(), libm::atan2(a, b).to_bits());
            prop_assert_eq!(DeterministicF64(a).powf(DeterministicF64(b)).0.to_bits(), libm::pow(a, b).to_bits());
        }
    }
}
