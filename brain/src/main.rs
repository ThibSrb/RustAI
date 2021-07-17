use brain::*;
//use std::fs::File;
//use std::io::Write;

fn main(){

    /*
    //XOR IA TESTING
    //let mut xorer = Brain::genetic_selection(2, 1, 3, 3,ActivationFunction::Heaviside ,Brain::learn_xor, 10000, 1000, 0.4, 0.4);
    //xorer.save("xorer.bin");
    let mut xorer = Brain::load("xorer.bin");

    
    let e1 = vec![0.0,0.0];
    let e2 = vec![1.0,0.0];
    let e3 = vec![0.0,1.0];
    let e4 = vec![1.0,1.0];

    let r1 = xorer.compute(&e1);
    let r2 = xorer.compute(&e2);
    let r3 = xorer.compute(&e3);
    let r4 = xorer.compute(&e4);

    //println!("{}\n",Brain::sigmoide(0.1));

    println!("Xorer({:?})={:?}",e1,r1);
    println!("Xorer({:?})={:?}",e2,r2);
    println!("Xorer({:?})={:?}",e3,r3);
    println!("Xorer({:?})={:?}",e4,r4);
    */


    /*
    //OR IA TESTING
    let mut orer = Brain::genetic_selection(2, 1, 2, 3,ActivationFunction::Heaviside ,Brain::learn_or, 1000, 1000, 0.4, 0.4);
    //orer.save("orer.bin");
    //let mut orer = Brain::load("orer.bin");

    
    let e1 = vec![0.0,0.0];
    let e2 = vec![1.0,0.0];
    let e3 = vec![0.0,1.0];
    let e4 = vec![1.0,1.0];

    let r1 = orer.compute(&e1);
    let r2 = orer.compute(&e2);
    let r3 = orer.compute(&e3);
    let r4 = orer.compute(&e4);

    //println!("{}\n",Brain::sigmoide(0.1));

    println!("Orer({:?})={:?}",e1,r1);
    println!("Orer({:?})={:?}",e2,r2);
    println!("Orer({:?})={:?}",e3,r3);
    println!("Orer({:?})={:?}",e4,r4);
    */

    
    //SORTITION TESTING AI TESTS
    //let mut sorterer = Brain::genetic_selection(8, 1, 4, 3, ActivationFunction::Heaviside, Brain::learn_sortition, 10, 10000, 0.3, 0.3);
    let mut sorterer = Brain::load("sorterer.bin");
    //sorterer.save("sorterer.bin");

    let mut l1 = Brain::data_scaler(&vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]);
    let mut l2 = Brain::data_scaler(&vec![3.0,2.0,1.0,4.0,6.0,8.0,7.0,5.0]);

    let lr1 = sorterer.compute(&mut l1);
    let lr2 = sorterer.compute(&mut l2);

    println!("Sortester({:?})={:?}",l1,lr1);
    println!("Sortester({:?})={:?}",l2,lr2);
    

    /*
    let data:Vec<f64> = vec![-2.0,-1.0,0.0,1.0,2.0,3.0,4.0];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));

    let data:Vec<f64> = vec![-4.0,-3.0,-2.0,-1.0,1.0,2.0];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));

    let data:Vec<f64> = vec![-4.0,-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));

    let data:Vec<f64> = vec![0.0,0.1];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));

    let data:Vec<f64> = vec![0.0,0.0];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));

    let data:Vec<f64> = vec![-1.0,1.0];
    println!("data_scaler({:?}) = {:?}", data, Brain::data_scaler(&data));
    */
    
}
