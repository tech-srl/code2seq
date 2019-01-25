package JavaExtractor.Common;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.File;

/**
 * This class handles the programs arguments.
 */
public class CommandLineValues {
    @Option(name = "--file", required = false)
    public File File = null;

    @Option(name = "--dir", required = false, forbids = "--file")
    public String Dir = null;

    @Option(name = "--max_path_length", required = true)
    public int MaxPathLength;

    @Option(name = "--max_path_width", required = true)
    public int MaxPathWidth;

    @Option(name = "--num_threads", required = false)
    public int NumThreads = 64;

    @Option(name = "--min_code_len", required = false)
    public int MinCodeLength = 1;

    @Option(name = "--max_code_len", required = false)
    public int MaxCodeLength = -1;

    @Option(name = "--max_file_len", required = false)
    public int MaxFileLength = -1;

    @Option(name = "--pretty_print", required = false)
    public boolean PrettyPrint = false;

    @Option(name = "--max_child_id", required = false)
    public int MaxChildId = 3;

    public CommandLineValues(String... args) throws CmdLineException {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            throw e;
        }
    }

    public CommandLineValues() {

    }
}