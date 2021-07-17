use graph::*;
use rand::Rng;

//SAVEFILE
extern crate savefile;
use savefile::prelude::*;
#[macro_use]
extern crate savefile_derive;

#[derive(Debug, Clone, Savefile)]
pub struct Brain {
    weights: Vec<f64>,
    values: Vec<f64>,
    network: Graph,
    pub layersizes: Vec<usize>,
    pub nbin: usize,
    pub nbout: usize,
    pub activation_function: ActivationFunction,
}

#[derive(Debug, Clone, Savefile, Copy)]
pub enum ActivationFunction {
    Sigmoide,
    Heaviside,
    Indentity,
}

impl Brain {
    pub fn new_basic(
        nbin: usize,
        nbout: usize,
        width: usize,
        depth: usize,
        activation_function: ActivationFunction,
    ) -> Brain {
        let mut rng = rand::thread_rng();
        let nb_vertex = nbin + nbout + (width * (depth - 2));
        let mut res: Brain = Brain {
            network: Graph::new(nb_vertex, false, false),
            weights: vec![0.0; nb_vertex],
            values: vec![0.0; nb_vertex],
            layersizes: vec![0; depth],
            nbin: nbin,
            nbout: nbout,
            activation_function: activation_function,
        };
        for i in 1..(depth - 1) {
            res.layersizes[i] = width;
        }
        res.layersizes[0] = nbin;
        res.layersizes[depth - 1] = nbout;

        let mut sl = 0; // start of layer
        let mut snl = nbin; // start of next layer
        let mut snnl = nbin + res.layersizes[1];
        for ls in 0..(depth - 1) {
            for n in sl..snl {
                for nn in snl..snnl {
                    res.network.add_edge(n, nn);
                    res.network
                        .set_cost((n, nn), (rng.gen::<f64>() * 2.0) - 1.0);
                    //res.network.add_edge(nn, n);
                }
            }
            sl += res.layersizes[ls];
            snl += res.layersizes[ls + 1];
            if ls + 2 < depth {
                snnl += res.layersizes[ls + 2];
            }
        }
        res
    }

    pub fn to_dot(&mut self) -> String {
        let mut labels: Vec<String> = vec![];
        for i in self.weights.iter() {
            labels.push((*i).to_string());
        }
        //self.network.to_dot_labeled(labels)
        self.network.to_dot()
    }

    pub fn heaviside(x: f64) -> f64 {
        match x {
            y if y <= 0.0 => 0.0,
            _ => 1.0,
        }
    }

    pub fn sigmoide(x: f64) -> f64 {
        1.0 / (1.0 + (-5.0 * x).exp())
    }

    pub fn compute(&mut self, entry: &Vec<f64>) -> Vec<f64> {
        let mut res = Vec::new();
        if self.nbin == entry.len() {
            let mut traversal = vec![];
            let mut depths = vec![];
            let mut mark = vec![false; self.network.order];
            //compute the entries
            for (cur, val) in entry.iter().enumerate() {
                self.values[cur] = *val;
                mark[cur] = true;
                self.values[cur] += self.weights[cur];
                self.values[cur] = self.values[cur];
                self.values[cur] = match self.activation_function {
                    ActivationFunction::Heaviside => Brain::heaviside(self.values[cur]),
                    ActivationFunction::Indentity => self.values[cur],
                    _ => Brain::sigmoide(self.values[cur]),
                };
                for i in self.network.adjlist[cur].iter() {
                    self.values[*i] += self.values[cur] * self.network.get_cost((cur, *i));
                    if !mark[*i] {
                        mark[*i] = true;
                        traversal.push(*i);
                        depths.push(0);
                    }
                }
            }
            //compute the traversal exept for the entries
            while !traversal.is_empty() {
                let cur = traversal[0];
                let depth = depths[0];
                depths.remove(0);
                traversal.remove(0);

                if cur < self.network.order - self.nbout {
                    self.values[cur] += self.weights[cur];
                    self.values[cur] = self.values[cur];
                    self.values[cur] = match self.activation_function {
                        ActivationFunction::Heaviside => Brain::heaviside(self.values[cur]),
                        ActivationFunction::Indentity => self.values[cur],
                        _ => Brain::sigmoide(self.values[cur]),
                    };
                    for j in self.layersizes[depth]..self.network.adjlist[cur].len() {
                        let i = &self.network.adjlist[cur][j];
                        self.values[*i] += self.values[cur] * self.network.get_cost((cur, *i));
                        if !mark[*i] {
                            mark[*i] = true;
                            traversal.push(*i);
                            depths.push(depth + 1);
                        }
                    }
                } else {
                    self.values[cur] += self.weights[cur];
                }
            }
            for (i, _j) in (0..self.network.order).rev().zip(0..self.nbout) {
                res.push(self.values[i]);
            }
        } else {
            eprintln!("Uncompatible entry");
        }
        self.values.iter_mut().for_each(|x| *x = 0.0); //resets all neurones values
        res.iter().rev().map(|x| f64::abs(*x)).collect()
    }

    pub fn apply_mutation(&mut self, prob: f64, factor: f64) {
        let mut rng = rand::thread_rng();
        let mut alea: f64;
        let mut mutation: f64;
        let mut mark = vec![false; self.network.order];
        for i in 0..self.network.order {
            mark[i] = true;
            alea = rng.gen::<f64>();
            if alea <= prob {
                mutation = (rng.gen::<f64>() * 2.0 - 1.0) * factor;
                self.weights[i] += mutation;
                for j in 0..self.network.adjlist[i].len() {
                    if !mark[j] {
                        alea = rng.gen::<f64>();
                        if alea <= prob {
                            mutation = (rng.gen::<f64>() * 2.0 - 1.0) * factor;
                            self.network
                                .set_cost((i, j), self.network.get_cost((i, j)) + mutation);
                        }
                    }
                }
            }
        }
    }

    pub fn genetic_selection<F>(
        nbin: usize,
        nbout: usize,
        width: usize,
        depth: usize,
        activation_function: ActivationFunction,
        compete_function: F,
        nb_gen: usize,
        gen_size: usize,
        mutation_prob: f64,
        mutation_fac: f64,
    ) -> Brain where F:Fn(&mut Vec<Brain>)->Brain{
        let mut gen = vec![];
        for _u in 0..gen_size {
            gen.push(Brain::new_basic(
                nbin,
                nbout,
                width,
                depth,
                activation_function,
            ));
        }
        for _i in 0..nb_gen {
            let best: Brain = compete_function(&mut gen);
            gen[0] = best.clone();
            let mut stop_index = gen_size;
            if gen_size >= 100{
                stop_index = gen_size - (gen_size/100);
                for j in stop_index..gen_size{
                    gen[j] = Brain::new_basic(nbin, nbout, width, depth, activation_function);
                }
            }
            for j in 1..stop_index {
                let mut wi = best.clone();
                wi.apply_mutation(mutation_prob, mutation_fac);
                gen[j] = wi;
            }
        }
        let best: Brain = compete_function(&mut gen);
        return best;
    }

    pub fn learn_xor(pool: &mut Vec<Brain>) -> Brain {
        //let mut xorer = Brain::genetic_selection(2, 1, 4, 3, Brain::learn_xor, 500, 5000, 0.4, 0.4); simple
        //let mut xorer = Brain::genetic_selection(2, 1, 5, 5, Brain::learn_xor, 500, 5000, 0.4, 0.4); more efficient but need a greater time of learn
        let mut error_values = vec![0.0; pool.len()];
        for (i, brain) in pool.iter_mut().enumerate() {
            let r1 = brain.compute(&vec![0.0, 0.0]);
            let r2 = brain.compute(&vec![0.0, 1.0]);
            let r3 = brain.compute(&vec![1.0, 0.0]);
            let r4 = brain.compute(&vec![1.0, 1.0]);

            error_values[i] = f64::abs(0.0 - r1[0])
                + f64::abs(1.0 - r2[0])
                + f64::abs(1.0 - r3[0])
                + f64::abs(0.0 - r4[0]);
        }
        //get the minimum error
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        pool[bi].clone()
    }

    pub fn learn_or(pool: &mut Vec<Brain>) -> Brain {
        //let mut xorer = Brain::genetic_selection(2, 1, 4, 3, Brain::learn_xor, 500, 5000, 0.4, 0.4); simple
        //let mut xorer = Brain::genetic_selection(2, 1, 5, 5, Brain::learn_xor, 500, 5000, 0.4, 0.4); more efficient but need a greater time of learn
        let mut error_values = vec![0.0; pool.len()];
        for (i, brain) in pool.iter_mut().enumerate() {
            let r1 = brain.compute(&vec![0.0, 0.0]);
            let r2 = brain.compute(&vec![0.0, 1.0]);
            let r3 = brain.compute(&vec![1.0, 0.0]);
            let r4 = brain.compute(&vec![1.0, 1.0]);

            error_values[i] = f64::abs(0.0 - r1[0])
                + f64::abs(1.0 - r2[0])
                + f64::abs(1.0 - r3[0])
                + f64::abs(1.0 - r4[0]);
        }
        //get the minimum error
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        pool[bi].clone()
    }

    pub fn learn_sortition(pool: &mut Vec<Brain>) -> Brain {
        let mut error_values = vec![0.0; pool.len()];
        for _i in 0..100 {
            let expt: f64;
            let mut rng = rand::thread_rng();
            let mut alea: Vec<u64> = (0..pool[0].nbin).map(|_x| rng.gen_range(0..10)).collect();
            if rng.gen::<f64>() > 0.5 {
                alea.sort();
            }
            if alea.windows(2).all(|w| w[0] <= w[1]) {
                expt = 1.0;
            } else {
                expt = 0.0;
            }
            //println!("Alea = {:?}, Expt = {:?}", alea, expt);
            for (i, brain) in pool.iter_mut().enumerate() {
                let l = Brain::data_scaler(&alea.iter().map(|x| *x as f64).collect());
                let r = brain.compute(&l);
                error_values[i] += f64::abs(expt - r[0]);
            }
        }
        //get the minimum error
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        println!("error : {}", error_values[bi]);
        pool[bi].clone()
    }

    pub fn save(&self, path: &str) {
        save_file(path, 0, self).unwrap()
    }

    pub fn load(path: &str) -> Brain {
        load_file(path, 0).unwrap()
    }

    pub fn max_vf64(target: &[f64]) -> f64 {
        let mut res = f64::NEG_INFINITY;
        for i in target.iter() {
            if *i > res {
                res = *i;
            }
        }
        res
    }

    pub fn min_vf64(target: &[f64]) -> f64 {
        let mut res = f64::INFINITY;
        for i in target.iter() {
            if *i < res {
                res = *i;
            }
        }
        res
    }

    pub fn data_scaler(target: &Vec<f64>) -> Vec<f64> {
        let mut res = target.clone();
        let mut min: f64 = Brain::min_vf64(target);
        let mut max: f64 = Brain::max_vf64(target);
        if min != max {
            let mut tweak = (0.0 - min) / (max + (0.0 - min));
            if max < 0.0 {
                min = f64::abs(min);
                max = f64::abs(max);
                tweak = (0.0 - min) / (max + (0.0 - min));
                res.iter_mut().for_each(|x| {
                    *x = -1.0 * ((f64::abs(*x) + (0.0 - min)) / (max + (0.0 - min)) - tweak)
                });
            } else {
                res.iter_mut()
                    .for_each(|x| *x = (*x + (0.0 - min)) / (max + (0.0 - min)) - tweak);
            }
        } else {
            res.iter_mut().for_each(|x| {
                if *x != 0.0 {
                    *x = *x / f64::abs(*x);
                } else {
                    *x = 0.0;
                }
            });
        }
        res
    }
}
