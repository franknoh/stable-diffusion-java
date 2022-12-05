package org.franknoh;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

class Main {
    public static void main(String[] args) {
        Device device;
        if(Engine.getInstance().getGpuCount() > 0) {
        device = Device.gpu();
        } else {
        device = Device.cpu();
        }
        NDManager manager = NDManager.newBaseManager(device);
        Generator generator = new Generator();
        Tokenizer tokenizer = new Tokenizer();
        Clip clip = new Clip(tokenizer);
        Diffusion diffusion = new Diffusion();
        Decoder decoder = new Decoder();

        String prompt = "A photo of an astronaut riding a horse.";
        String uncond_prompt = "";
        int n_interface_steps = 10;
        int height = 512;
        int width = 512;
        float cfg_scale = 7.5f;

        NDArray cond_clip_output = clip.embedText(prompt);
        NDArray uncond_clip_output = clip.embedText(uncond_prompt);

        System.out.println("cond_clip_output: " + cond_clip_output);
        System.out.println("uncond_clip_output: " + uncond_clip_output);

        NDArray context = manager.zeros(new Shape(2, 77, 768));
        context.set(new NDIndex(0), cond_clip_output);
        context.set(new NDIndex(1), uncond_clip_output);
        System.out.println("context: " + context);

        Sampler sampler = new Sampler(n_interface_steps);
        NDArray latents = generator.sample(new Shape(1, 4, height/8, width/8));
        latents = latents.mul(sampler.initial_scale);
        System.out.println("latents: " + latents);
        decoder.saveImage(latents.duplicate(), "initial");
        for (int i = 0; i < n_interface_steps; i++) {
            NDArray timestep = sampler.timesteps.get(new NDIndex(i));
            NDArray time_embedding = sampler.get_time_embedding(timestep);
            System.out.println("n_inference_steps: " + i);
            System.out.println("timestep: " + timestep);
            System.out.println("time_embedding: " + time_embedding);
            NDArray output = diffusion.run(latents.mul(sampler.get_input_scale()), context, time_embedding);
            NDArray output_cond = manager.zeros(new Shape(1, 4, height/8, width/8));
            output_cond.set(new NDIndex(0), output.get(new NDIndex(0)));
            NDArray output_uncond = manager.zeros(new Shape(1, 4, height/8, width/8));
            output_uncond.set(new NDIndex(0), output.get(new NDIndex(1)));
            output  = output_cond.sub(output_uncond).mul(cfg_scale).add(output_uncond);
            latents = sampler.step(latents, output);
            System.out.println("latents: " + latents);
            decoder.saveImage(latents.duplicate(), "step_" + i);
        }
        decoder.saveImage(latents.duplicate(), "final");
    }
}