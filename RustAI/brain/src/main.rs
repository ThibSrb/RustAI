use brain::*;
//use std::fs::File;
//use std::io::Write;

fn main(){
    /*
    let mut b = Brain::new_basic(2, 2, 4, 4);

    let mut compute_entry:Vec<f64> = vec![2.3,1.5];
    let computed_output:Vec<f64> = b.compute_entry(&mut compute_entry);
    //println!("{}",b.to_dot());
    println!("Entry to compute : {:?}\nComputed output : {:?}\n",compute_entry, computed_output);
    b.apply_mutation(0.2, 0.2);
    let computed_output:Vec<f64> = b.compute_entry(&mut compute_entry);
    println!("Entry to compute : {:?}\nComputed output : {:?}",compute_entry, computed_output);
    */
    let mut xorer = Brain::genetic_selection(2, 1, 5, 5, Brain::learn_xor, 500, 5000, 0.4, 0.4);
    
    let mut e1 = vec![0.0,0.0];
    let mut e2 = vec![1.0,0.0];
    let mut e3 = vec![0.0,1.0];
    let mut e4 = vec![1.0,1.0];

    let r1 = xorer.compute_entry(&mut e1);
    let r2 = xorer.compute_entry(&mut e2);
    let r3 = xorer.compute_entry(&mut e3);
    let r4 = xorer.compute_entry(&mut e4);

    println!("{}",xorer.to_dot());

    println!("Xorer({:?})={:?}",e1,r1);
    println!("Xorer({:?})={:?}",e2,r2);
    println!("Xorer({:?})={:?}",e3,r3);
    println!("Xorer({:?})={:?}",e4,r4);

}
