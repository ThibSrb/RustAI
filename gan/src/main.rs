#![allow(dead_code, unused)]
use brain::*;
use rand::Rng;
use std::thread;
use std::time;

fn alea_word(nbl: usize) -> Vec<char> {
    let mut rng = rand::thread_rng();
    let chars = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];
    let mut resstr = vec![' '; nbl];
    for i in 0..(nbl - rng.gen_range(0..(nbl - 2))) {
        resstr[i] = chars[rng.gen_range(0..26) as usize];
    }
    resstr
}

fn get_alea_words(nbl: usize, nb: usize) -> Vec<Vec<char>> {
    let mut res: Vec<Vec<char>> = vec![];
    for _i in 0..nb {
        res.push(alea_word(nbl));
    }
    res
}

fn get_true_words(nbl: usize, nb: usize, path: &str) -> Vec<Vec<char>> {
    let mut rng = rand::thread_rng();
    let mut res: Vec<Vec<char>> = vec![];
    let content = std::fs::read_to_string(path).unwrap();
    let table: Vec<Vec<char>> = content
        .split('\n')
        .map(|x| x.to_lowercase().chars().collect())
        .collect();
    let size = table.len();
    let mut alea;
    for _i in 0..nb {
        alea = rng.gen_range(0..size);
        while (table[alea].len() > nbl + 1) {
            alea = rng.gen_range(0..size);
        }
        let mut preres = table[alea].clone();
        preres.pop();
        while preres.len() < nbl {
            preres.push(' ');
        }
        res.push(preres);
    }
    res
}

fn learn_words(pool: &mut Vec<Brain>, path: &str, nb: usize) -> Vec<Brain> {
    let nb = 1024;
    let len = pool.len();
    let reals = get_true_words(pool[0].nbin, nb / 2, path);
    let fakes = get_alea_words(pool[0].nbin, nb / 2);
    let mut error_values: Vec<f64> = vec![0.0; len];
    let mut w: Vec<char>;
    let mut expt: f64;

    for iw in 0..nb {
        if iw % 2 == 0 {
            w = reals[iw / 2 as usize].clone();
            expt = 1.0;
        } else {
            w = fakes[iw / 2 as usize].clone();
            expt = 0.0
        }
        //println!("word : {:?}, expected result : {:?}",w.iter().collect::<String>(),expt);
        for i in 0..len {
            error_values[i] += f64::abs(expt - pool[i].compute(&letters_to_entry(&w))[0]);
            //error_values[i] += 1.0;
        }
    }

    //get the minimum error
    let mut res = vec![];
    for i in 0..nb {
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        if i == 0 {
            println!("error : {}", error_values[bi]);
        }
        res.push(pool[bi].clone());
        pool.remove(bi);
        error_values.remove(bi);
    }
    res
}

fn multithreaded_learn_words(pool: &mut Vec<Brain>, path: &str, nb_b: usize) -> Vec<Brain> {
    let nb = 128;
    let len = pool.len();
    //println!("len = {}", len);
    let mut error_values: Vec<f64> = vec![0.0; len];
    let mut threads = vec![];
    let nbth = 8;
    for ths in 0..nbth {
        let pa = path.clone().to_string();
        let mut errs = vec![0.0; len];
        let mut brains = pool.clone();
        let nth = nbth;

        let th = thread::spawn(move || {
            let reals = get_true_words(brains[0].nbin, nb / (2 * nth), &pa);
            let fakes = get_alea_words(brains[0].nbin, nb / (2 * nth));
            let mut w: Vec<char>;
            let mut expt = 0.0;

            for iw in 0..(nb / nth) {
                if iw % 2 == 0 {
                    w = reals[iw / 2 as usize].clone();
                    expt = 1.0;
                } else {
                    w = fakes[iw / 2 as usize].clone();
                    expt = 0.0
                }
                for i in 0..len {
                    errs[i] += f64::abs(expt - brains[i].compute(&letters_to_entry(&w))[0]);
                    //errs[i] += 1.0;
                }
            }
            errs
        });
        threads.push(th);
    }
    let e = 0;
    for th in threads {
        let e = th.join().unwrap();
        for i in 0..len {
            error_values[i] += e[i];
        }
    }
    //get the minimum error
    let mut res = vec![];
    for i in 0..nb_b {
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        if i == 0 {
            println!("error : {}", error_values[bi]);
        }
        res.push(pool[bi].clone());
        pool.remove(bi);
        error_values.remove(bi);
    }
    res
}

fn learn_against(pool: &mut Vec<Brain>, adversary: &mut Brain) -> Brain {
    pool[0].clone()
}

fn letters_to_entry(letters: &Vec<char>) -> Vec<f64> {
    let mut res = vec![];
    for i in letters.iter() {
        match *i {
            ' ' => res.push(1.0),
            _ => res.push((*i as u8 as f64 - 'a' as u8 as f64) / (123 as f64 - 'a' as u8 as f64)) ,
        }
    }
    res
}

fn main() {
    let compete_an = |pool: &mut Vec<Brain>, nb: usize| -> Vec<Brain> {
        multithreaded_learn_words(pool, "PrenomsFRUTF", nb)
        //learn_words(pool, "PrenomsFRUTF", nb)
    };
    let mut an = Brain::genetic_selection(
        8,
        1,
        4,
        3,
        ActivationFunction::Heaviside,
        compete_an,
        10,
        100000,
        0.005,
        1.0,
    ); //adversarial network
    an.save("advNet5.bin");
    //let mut an = Brain::load("advNet4.bin");
    let t1: Vec<char> = get_true_words(8, 1, "PrenomsFRUTF")[0].clone();
    let t2: Vec<char> = get_alea_words(8, 1)[0].clone();
    let r1 = an.compute(&letters_to_entry(&t1));
    let r2 = an.compute(&letters_to_entry(&t2));

    println!("an({:?}) = {:?}", t1.iter().collect::<String>(), r1);
    println!("an({:?}) = {:?}", t2.iter().collect::<String>(), r2);
    /*
    let compete_gn = |pool:&mut Vec<Brain>| -> Brain {
        let mut u = an.clone();
        learn_against(pool, &mut u)
    };
    let mut gn = Brain::genetic_selection(1, 10, 5, 3, ActivationFunction::Heaviside,compete_gn, 100, 1000, 0.4, 0.4); //generative network
    */

    /*
    println!("{:?}\n", get_alea_words(10, 1));
    println!("{:?}\n", get_true_words(10, 1, "PrenomsFRUTF"));
    println!("{:?} = {:?}\n", 'b', ('b' as u8 as f64 - 'a' as u8 as f64) / ('z' as u8 as f64 - 'a' as u8 as f64));
    println!("{:?} = {:?}\n", 'z', ('z' as u8 as f64 - 'a' as u8 as f64) / ('z' as u8 as f64 - 'a' as u8 as f64));
    */
    /*
    let b = get_alea_words(8,1)[0].clone();
    let a = get_true_words(8, 1, "PrenomsFRUTF")[0].clone();
    println!("{:?} = {:?}",a,letters_to_entry(&a));
    println!("{:?} = {:?}",b,letters_to_entry(&b));
    */
}
