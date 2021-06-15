use std::collections::HashMap;

extern crate savefile;
use savefile::*;
#[macro_use]
extern crate savefile_derive;

#[derive(Debug, Clone, Savefile)]
pub struct Graph {
    pub order: usize,
    pub directed: bool,
    pub adjlist: Vec<Vec<usize>>,
    pub hascosts: bool,
    pub costs: HashMap<(usize, usize), f64>,
}

impl Graph {
    pub fn new(order: usize, directed: bool, hascosts: bool) -> Graph {
        let mut res: Graph = Graph {
            order: order,
            directed: directed,
            adjlist: Vec::new(),
            hascosts: hascosts,
            costs: HashMap::new(),
        };
        for _i in 00..order {
            res.adjlist.push(vec![]);
        }
        res
    }

    pub fn add_edge(&mut self, src: usize, dst: usize) {
        if src >= self.order || dst >= self.order {
            eprintln!("Error : src and dst must be strictly inferior than the graph order");
        } else {
            if !self.directed {
                if !self.adjlist[dst].contains(&src) {
                    self.adjlist[dst].push(src);
                }
            }
            if !self.adjlist[src].contains(&dst) {
                self.adjlist[src].push(dst);
            }
        }
    }

    pub fn get_cost(&self, edge: (usize, usize)) -> f64 {
        let res: f64;
        match self.costs.get(&edge) {
            Some(val) => res = *val,
            None => res = 0.0,
        }
        res
    }

    pub fn set_cost(&mut self, edge: (usize, usize), val: f64) {
        self.costs.insert(edge, val);
        if !self.directed {
            let (a, b) = edge;
            self.costs.insert((b, a), val);
        }
    }

    pub fn add_vertex(&mut self, number: u64) {
        for _i in 0..number {
            self.order += 1;
            self.adjlist.push(Vec::new());
        }
    }

    pub fn remove_edge(&mut self, src: usize, dst: usize) {
        if src >= self.order || dst >= self.order {
            eprintln!("Error : src and dst must be strictly inferior than the graph order");
        } else {
            if !self.directed {
                if let Some(ind2) = self.adjlist[dst].iter().position(|x| *x == src) {
                    self.adjlist[dst].remove(ind2);
                }
            }
            if let Some(ind1) = self.adjlist[src].iter().position(|x| *x == dst) {
                self.adjlist[src].remove(ind1);
            }
        }
    }

    pub fn to_dot(&mut self) -> String {
        let link: &str;
        let mut dot: String;
        if self.directed {
            dot = String::from("digraph {\n");
            link = " -> ";
        } else {
            dot = String::from("graph G {\n");
            link = " -- ";
        }
        for i in 0..self.order {
            dot += &format!(" {}\n", i);
            for j in self.adjlist[i].iter() {
                let cost: String;
                if self.hascosts {
                    cost = format!(" [label={}]", self.get_cost((i, *j)));
                } else {
                    cost = String::from("");
                }
                if self.directed || *j <= i {
                    dot += &format!(" {}{}{}{}\n", i, link, j, cost);
                }
            }
        }
        dot += "}";
        dot
    }

    pub fn to_dot_labeled(&mut self, label: Vec<String>) -> String {
        let link: &str;
        let mut dot: String;
        if self.directed {
            dot = String::from("digraph {\n");
            link = " -> ";
        } else {
            dot = String::from("graph G {\n");
            link = " -- ";
        }
        for i in 0..self.order {
            dot += &format!(" {} [label={}]\n", i, label[i]);
            for j in self.adjlist[i].iter() {
                let cost: String;
                if self.hascosts {
                    cost = format!(" [label={}]", self.get_cost((i, *j)));
                } else {
                    cost = String::from("");
                }
                if self.directed || *j <= i {
                    dot += &format!(" {}{}{}{}\n", i, link, j, cost);
                }
            }
        }
        dot += "}";
        dot
    }

    pub fn save(&self,path:&str){
        save_file(path, 0, self).unwrap()
    }

    pub fn load(path:&str)->Graph{
        load_file(path, 0).unwrap()
    }
}
