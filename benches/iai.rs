use imgref::{Img, ImgRefMut};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

static mut BUFFER: [u32; WIDTH * HEIGHT] = [0; WIDTH * HEIGHT];

#[inline(always)]
fn img() -> ImgRefMut<'static, u32> {
    Img::new(unsafe { &mut BUFFER[..] }, WIDTH, HEIGHT)
}

fn blur_argb_16() {
    slur::blur_argb(&mut img(), 16)
}
fn blur_argb_128() {
    slur::blur_argb(&mut img(), 128)
}
fn blur_argb_1024() {
    slur::blur_argb(&mut img(), 1024)
}

// parallel versions are non-deterministic

iai::main!(blur_argb_16, blur_argb_128, blur_argb_1024,);
