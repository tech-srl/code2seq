package JavaExtractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import org.kohsuke.args4j.CmdLineException;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.FeaturesEntities.ProgramRelation;

import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;

public class App {
	private static CommandLineValues s_CommandLineValues;

	public static void main(String[] args) {
		try {
			s_CommandLineValues = new CommandLineValues(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
			return;
		}

		if (s_CommandLineValues.File != null) {
			ExtractFeaturesTask extractFeaturesTask = new ExtractFeaturesTask(s_CommandLineValues,
					s_CommandLineValues.File.toPath());
			extractFeaturesTask.processFile();
		} else if (s_CommandLineValues.Dir != null) {
			extractDir();
		}
	}

	private static void extractDir() {
		List<Path> files = null;
		try {
			files = Files.walk(Paths.get(s_CommandLineValues.Dir)).filter(Files::isRegularFile)
					.filter(p -> p.toString().toLowerCase().endsWith(".java")).collect(toList());
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		int[] count = new int[1];
		final int CHUNK_SIZE = 100;
		Map<Integer, List<Path>> batchedFiles = files.stream().collect( Collectors.groupingBy(
				file -> {
					count[0]++;
					return Math.floorDiv(count[0], CHUNK_SIZE);
				} )
		);

		for (List<Path> filesBatch : batchedFiles.values()) {
			ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
			LinkedList<ExtractFeaturesTask> tasks = new LinkedList<>();
			filesBatch.forEach(f -> {
				ExtractFeaturesTask task = new ExtractFeaturesTask(s_CommandLineValues, f);
				tasks.add(task);
			});

			try {
				executor.invokeAll(tasks);
			} catch (InterruptedException e) {
				e.printStackTrace();
			} finally {
				executor.shutdown();
			}
		}
	}
}
