use graph::*;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Brain {
    weights: Vec<f64>,
    values: Vec<f64>,
    network: Graph,
    pub layersizes: Vec<usize>,
    nbin: usize,
    nbout: usize,
}

impl Brain {
    pub fn new_basic(nbin: usize, nbout: usize, width: usize, depth: usize) -> Brain {
        let mut rng = rand::thread_rng();
        let nb_verrtex = nbin + nbout + (width * (depth - 2));
        let mut res: Brain = Brain {
            network: Graph::new(nb_verrtex, false, false),
            weights: vec![0.0; nb_verrtex],
            values: vec![0.0; nb_verrtex],
            layersizes: vec![0; depth],
            nbin: nbin,
            nbout: nbout,
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
            y if y < 0.0 => 0.0,
            _ => 1.0,
        }
    }

    pub fn compute_entry(&mut self, entry: &mut Vec<f64>) -> Vec<f64> {
        let mut res = Vec::new();
        if self.nbin == entry.len() {
            let mut traversal = vec![];
            let mut mark = vec![false; self.network.order];

            for (i, val) in entry.iter().enumerate() {
                self.values[i] = *val;
                traversal.push(i);
                mark[i] = true;
            }

            while !traversal.is_empty() {
                let cur = traversal[0];
                traversal.remove(0);
                if cur < self.network.order - self.nbout {
                    self.values[cur] += self.weights[cur];
                    self.values[cur] = Brain::heaviside(self.values[cur]);
                    for i in self.network.adjlist[cur].iter() {
                        self.values[*i] += self.values[cur] * self.network.get_cost((cur, *i));
                        if !mark[*i] {
                            mark[*i] = true;
                            traversal.push(*i);
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
        res.iter().rev().map(|x| *x).collect()
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
        compete_function: F,
        nb_gen: usize,
        gen_size: usize,
        mutation_prob: f64,
        mutation_fac: f64,
    ) -> Brain
    where
        F: Fn(&mut Vec<Brain>) -> Brain,
    {
        let mut gen = vec![];
        for _u in 0..gen_size {
            gen.push(Brain::new_basic(nbin, nbout, width, depth));
        }
        for _i in 0..nb_gen {
            let best: Brain = compete_function(&mut gen);
            gen[0] = best.clone();
            for j in 1..gen_size {
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
            let r1 = brain.compute_entry(&mut vec![0.0, 0.0]);
            let r2 = brain.compute_entry(&mut vec![0.0, 1.0]);
            let r3 = brain.compute_entry(&mut vec![1.0, 0.0]);
            let r4 = brain.compute_entry(&mut vec![1.0, 1.0]);

            error_values[i] = f64::abs(0.0 - r1[0])
                + f64::abs(1.0 - r2[0])
                + f64::abs(1.0 - r3[0])
                + f64::abs(0.0 - r4[0]);
        }
        let mut min = -1.0;
        let mut bi = 0;
        for (i, j) in error_values.iter().enumerate() {
            if *j < min || min == -1.0 {
                min = *j;
                bi = i;
            }
        }
        //println!("Min of {:?} is {} at {}\n",error_values,error_values[bi],bi);
        pool[bi].clone()
    }
}
