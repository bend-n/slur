use crate::StackBlurrable;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

mod serial;
mod simd;

pub use serial::BlurU32;
pub use simd::u32xN;

use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Argb<T: StackBlurrable>([T; 4]);

impl From<u32> for Argb<BlurU32> {
    fn from(argb: u32) -> Self {
        let [a, r, g, b] = argb.to_be_bytes();
        let cvt = |i| BlurU32(i as u32);
        Self([cvt(a), cvt(r), cvt(g), cvt(b)])
    }
}

impl From<Argb<BlurU32>> for u32 {
    fn from(Argb([a, r, g, b]): Argb<BlurU32>) -> Self {
        let cvt = |i: BlurU32| i.0 as u8;
        u32::from_be_bytes([cvt(a), cvt(r), cvt(g), cvt(b)])
    }
}

impl<const N: usize> From<[u32; N]> for Argb<u32xN<N>>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn from(values: [u32; N]) -> Self {
        let arrs: [[u8; 4]; N] = values.map(u32::to_be_bytes);
        Self([
            u32xN(Simd::from_array(arrs.map(|a| a[0] as u32))),
            u32xN(Simd::from_array(arrs.map(|a| a[1] as u32))),
            u32xN(Simd::from_array(arrs.map(|a| a[2] as u32))),
            u32xN(Simd::from_array(arrs.map(|a| a[3] as u32))),
        ])
    }
}

impl<const N: usize> From<Argb<u32xN<N>>> for [u32; N]
where
    LaneCount<N>: SupportedLaneCount,
{
    fn from(value: Argb<u32xN<N>>) -> Self {
        let [a, r, g, b] = value.0.map(|i| i.0.to_array());
        std::array::from_fn(|i| u32::from_be_bytes([a[i], r[i], g[i], b[i]].map(|x| x as u8)))
    }
}

impl<T: StackBlurrable> Add for Argb<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: StackBlurrable> Sub for Argb<T> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: StackBlurrable> AddAssign for Argb<T> {
    fn add_assign(&mut self, rhs: Self) {
        let [a, r, g, b] = rhs.0;
        self.0[0] += a;
        self.0[1] += r;
        self.0[2] += g;
        self.0[3] += b;
    }
}

impl<T: StackBlurrable> SubAssign for Argb<T> {
    fn sub_assign(&mut self, rhs: Self) {
        let [a, r, g, b] = rhs.0;
        self.0[0] -= a;
        self.0[1] -= r;
        self.0[2] -= g;
        self.0[3] -= b;
    }
}

impl<T: StackBlurrable> Mul<usize> for Argb<T> {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        let [a, r, g, b] = self.0;
        Self([a * rhs, r * rhs, g * rhs, b * rhs])
    }
}

impl<T: StackBlurrable> Div<usize> for Argb<T> {
    type Output = Self;
    fn div(self, rhs: usize) -> Self::Output {
        let [a, r, g, b] = self.0;
        Self([a / rhs, r / rhs, g / rhs, b / rhs])
    }
}
