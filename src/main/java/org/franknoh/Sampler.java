package org.franknoh;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class Sampler {
    public float initial_scale;
    public int n_inference_steps;
    public int n_training_steps;
    public int lms_order;
    public int step_count;
    public NDArray timesteps;
    public NDArray sigmas;
    public NDList outputs;
    private final NDManager manager;

    Sampler(int n_inference_steps) {
        this.n_inference_steps = n_inference_steps;
        this.n_training_steps = 1000;
        this.lms_order =4;
        Device device;
        if(Engine.getInstance().getGpuCount() > 0) {
            device = Device.gpu();
        } else {
            device = Device.cpu();
        }
        this.manager = NDManager.newBaseManager(device);
        this.timesteps = this.manager.linspace(this.n_training_steps-1, 0, n_inference_steps);
        NDArray alphas_cumprod = get_alphas_cumprod(this.n_training_steps);
        NDArray sigmas = this.manager.ones(new Shape(this.n_training_steps)).sub(alphas_cumprod).div(alphas_cumprod).sqrt();
        NDArray log_sigmas = sigmas.log();
        log_sigmas = interp(this.timesteps, this.manager.arange(this.n_training_steps), log_sigmas);
        NDArray sigmas_diff = this.manager.zeros(new Shape(n_inference_steps+1));
        for (int i = 0; i < n_inference_steps; i++) {
            sigmas_diff.set(new NDIndex(i), log_sigmas.get(new NDIndex(i)).exp());
        }
        sigmas_diff.set(new NDIndex(n_inference_steps), 0);
        this.sigmas = sigmas_diff;
        this.initial_scale = 0.0f;
        for (int i = 0; i < this.n_inference_steps; i++) {
            this.initial_scale = Math.max(this.initial_scale, this.sigmas.get(i).getFloat());
        }
        this.step_count = 0;
        this.outputs = new NDList();
    }

   NDArray step(NDArray latents, NDArray output) {
       int t = this.step_count;
       this.step_count = this.step_count + 1;
       this.outputs.add(0, output);
       while (this.outputs.size() > this.lms_order-1) {
           this.outputs.remove(this.outputs.size()-1);
       }
       int order = this.outputs.size();
       for (int i = 0; i < order; i++) {
           NDArray t_output = this.outputs.get(i);
           NDArray x = this.manager.linspace(this.sigmas.getFloat(t), this.sigmas.getFloat(t+1), 81);
           NDArray y = this.manager.ones(new Shape(81));
           for (int j = 0; j < order; j++) {
               if (i == j) {
                   continue;
               }
               y = y.mul(x.sub(this.sigmas.get(t - j)));
               y = y.div(this.sigmas.get(t - i).sub(this.sigmas.get(t - j)));
           }
           float lms_coeff = trapz(y, x);
           latents = t_output.mul(lms_coeff).add(latents);
       }
       return latents;
    }

    float get_input_scale() {
        float sigma = this.sigmas.getFloat(this.step_count);
        return (float) Math.sqrt(1 / (sigma * sigma + 1));
    }

    NDArray get_alphas_cumprod(int n_training_steps){
        float beta_start = 0.00085f;
        float beta_end = 0.0120f;
        NDArray betas = this.manager.linspace((float) Math.sqrt(beta_start), (float) Math.sqrt(beta_end), n_training_steps);
        NDArray alphas = manager.ones(new Shape(n_training_steps)).sub(betas.mul(betas));
        NDArray alphas_cumprod = manager.ones(new Shape(n_training_steps));
        alphas_cumprod.set(new NDIndex(0), alphas.getFloat(0));
        for (int i = 1; i < n_training_steps; i++) {
            alphas_cumprod.set(new NDIndex(i), alphas_cumprod.getFloat(i-1) * alphas.getFloat(i));
        }
        return alphas_cumprod;
    }

    NDArray interp(NDArray x, NDArray xp, NDArray fp) {
        NDArray y = this.manager.zeros(new Shape(x.size()));
        x = x.toType(DataType.FLOAT32, false);
        xp = xp.toType(DataType.FLOAT32, false);
        fp = fp.toType(DataType.FLOAT32, false);
        for (int i = 0; i < x.size(); i++) {
            float x_val = x.getFloat(i);
            for(int j = 0; j < xp.size()-1; j++) {
                if(x_val >= xp.getFloat(j) && x_val <= xp.getFloat(j+1)) {
                    float fp_val = fp.getFloat(j);
                    float fp_next_val = fp.getFloat(j+1);
                    float xp_val = xp.getFloat(j);
                    float xp_next_val = xp.getFloat(j+1);
                    float y_val = fp_val + (fp_next_val - fp_val) * (x_val - xp_val) / (xp_next_val - xp_val);
                    y.set(new NDIndex(i), y_val);
                }
            }
        }
        return y;
    }

    float trapz(NDArray y, NDArray x){
        float sum = 0;
        for (int i = 0; i < 80; i++) {
            sum += (y.getFloat(i) + y.getFloat(i+1)) * (x.getFloat(i+1) - x.getFloat(i)) / 2;
        }
        return sum;
    }

    NDArray get_time_embedding(NDArray timestep) {
        NDArray freqs = this.manager.ones(new Shape(160)).mul(10000).pow(this.manager.arange(0, 160).div(160).mul(-1));
        NDArray x = timestep.mul(freqs);
        NDArray cos = x.cos();
        NDArray sin = x.sin();
        NDArray time_embedding = cos.concat(sin, 0);
        return time_embedding.reshape(new Shape(1, 320));
    }
}
