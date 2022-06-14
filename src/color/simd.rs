use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

pub use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct u32xN<const N: usize>(pub Simd<u32, N>)
where
    LaneCount<N>: SupportedLaneCount;

impl<const N: usize> Add for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const N: usize> Sub for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const N: usize> AddAssign for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<const N: usize> SubAssign for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const N: usize> Mul<usize> for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        Self(self.0 * Simd::<u32, N>::splat(rhs as u32))
    }
}

impl<const N: usize> Div<usize> for u32xN<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        Self(Simd::<u32, N>::from_array(
            self.0.to_array().map(|e| (e as usize / rhs) as u32),
        ))
    }
}
