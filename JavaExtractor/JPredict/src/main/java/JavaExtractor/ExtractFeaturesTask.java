package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import com.google.gson.Gson;

class ExtractFeaturesTask implements Callable<Void> {
    private final CommandLineValues commandLineValues;
    private final Path filePath;

    public ExtractFeaturesTask(CommandLineValues commandLineValues, Path path) {
        this.commandLineValues = commandLineValues;
        this.filePath = path;
    }

    @Override
    public Void call() {
        processFile();
        return null;
    }

    public void processFile() {
        ArrayList<ProgramFeatures> features;
        try {
            features = extractSingleFile();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        if (features == null) {
            return;
        }

        String toPrint = featuresToString(features);
        if (toPrint.length() > 0) {
            System.out.println(toPrint);
        }
    }

    private ArrayList<ProgramFeatures> extractSingleFile() throws IOException {
        String code;

        if (commandLineValues.MaxFileLength > 0 &&
                Files.lines(filePath, Charset.defaultCharset()).count() > commandLineValues.MaxFileLength) {
            return new ArrayList<>();
        }
        try {
            code = new String(Files.readAllBytes(filePath));
        } catch (IOException e) {
            e.printStackTrace();
            code = Common.EmptyString;
        }
        FeatureExtractor featureExtractor = new FeatureExtractor(commandLineValues, this.filePath);

        return featureExtractor.extractFeatures(code);
    }

    public String featuresToString(ArrayList<ProgramFeatures> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (ProgramFeatures singleMethodFeatures : features) {
            StringBuilder builder = new StringBuilder();

            String toPrint;
            if (commandLineValues.JsonOutput) {
                toPrint = new Gson().toJson(singleMethodFeatures);
            }
            else {
                toPrint = singleMethodFeatures.toString();
            }
            if (commandLineValues.PrettyPrint) {
                toPrint = toPrint.replace(" ", "\n\t");
            }
            builder.append(toPrint);


            methodsOutputs.add(builder.toString());

        }
        return StringUtils.join(methodsOutputs, "\n");
    }
}
