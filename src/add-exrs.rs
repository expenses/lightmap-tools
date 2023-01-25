use structopt::StructOpt;

#[derive(StructOpt)]
pub struct Opts {
    direct: std::path::PathBuf,
    indirect: std::path::PathBuf,
}

fn main() {
    let opts = Opts::from_args();

    let mut base = image::open(&opts.direct).unwrap().to_rgb32f();

    // Direct lighting seems to be too bright at the moment.
    for value in &mut *base {
        *value *= std::f32::consts::FRAC_1_PI * std::f32::consts::FRAC_1_PI;
    }

    let indirect = image::open(&opts.indirect).unwrap().to_rgb32f();

    for (base, new) in (&mut *base).iter_mut().zip(&*indirect) {
        *base += *new;
    }

    base.save("added.exr").unwrap();
}
