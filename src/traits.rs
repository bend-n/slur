//! The home of [`StackBlurrable`].

use std::ops::{Add, AddAssign, Div, Mul, SubAssign};

/// The trait for types which can be blurred by [`StackBlur`][crate::StackBlur].
///
/// This trait is auto-implemented for all types that satisfy its requirements.
///
/// Types that wish to implement this trait should be signed or use explicitly
/// wrapping arithmetic.
///
/// They should have a significantly higher precision than the pixel format that
/// they represent, as they may be multiplied by hundreds or thousands before
/// being divided. They should also ideally be `Copy` so that cloning is cheap.
pub trait StackBlurrable:
    Default
    + Copy
    + Add<Output = Self>
    + AddAssign
    + SubAssign
    + Mul<usize, Output = Self>
    + Div<usize, Output = Self>
{
}

impl<
        T: Default
            + Copy
            + Add<Output = T>
            + AddAssign
            + SubAssign
            + Mul<usize, Output = T>
            + Div<usize, Output = T>,
    > StackBlurrable for T
{
}
