use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct BlurU32(pub u32);

impl Add for BlurU32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for BlurU32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl AddAssign for BlurU32 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl SubAssign for BlurU32 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

impl Mul<usize> for BlurU32 {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        Self(self.0.wrapping_mul(rhs as u32))
    }
}

impl Div<usize> for BlurU32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: usize) -> Self::Output {
        Self(self.0.wrapping_div(rhs as u32))
    }
}
