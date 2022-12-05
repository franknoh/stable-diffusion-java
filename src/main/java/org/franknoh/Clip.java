package org.franknoh;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

public class Clip {
    private ZooModel clip;
    private Device device;
    private Tokenizer tokenizer;
    private NDManager manager;
    private Predictor<NDList, NDList> clip_predictor;
    Clip(Tokenizer tokenizer) {
        if(Engine.getInstance().getGpuCount() > 0) {
            this.device = Device.gpu();
        } else {
            this.device = Device.cpu();
        }
        this.manager = NDManager.newBaseManager(this.device);
        Criteria<NDList, NDList> clip_c = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get("src/main/resources/clip.pt"))
                .optEngine("PyTorch")
                .optDevice(this.device)
                .build();
        try {
            this.clip = ModelZoo.loadModel(clip_c);
        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
        this.tokenizer = tokenizer;
        this.clip_predictor = this.clip.newPredictor();
    }
    public NDArray run(List<Integer> fi_tokens) {
        NDList clip_input = new NDList();
        int[] tokens_array = new int[fi_tokens.size()];
        for (int i = 0; i < fi_tokens.size(); i++) {
            tokens_array[i] = fi_tokens.get(i);
        }
        int[][] tokens_array_2d = new int[][]{tokens_array};
        NDArray tokens_ndarray = this.manager.create(tokens_array_2d);
        clip_input.add(tokens_ndarray);
        NDList clip_output;
        try {
            clip_output = clip_predictor.predict(clip_input);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        return clip_output.get(0);
    }

    public NDArray embedText(String fi_text) {
        List<Integer> fi_tokens = this.tokenizer.encode(fi_text);
        return run(fi_tokens);
    }
}
