package tool;

import model.IndexNode;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public enum ProcessTool {
    INSTANCE;

    /**
     * Create mock index hierarchy nodes for test use
     *
     * @return
     */
    int numFrames ;
    ArrayList<double[][][][]> histograms = new ArrayList<>();
    ArrayList<Double> cummulativeDiffs = new ArrayList<>();



    ArrayList<Integer> keyFrameID = new ArrayList<Integer>();
    public List<IndexNode> getMockIndexNodes(ArrayList<Integer[]> data) {
        List<IndexNode> nodes = new ArrayList<>();
        int sencecount = 1;
        int shotcount = 1;
        int subshotcount = 1;

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i)[0] == 1) {
                IndexNode scene = new IndexNode("Scene" + Integer.toString(sencecount), data.get(i)[1] / 30.0);
                shotcount = 1;
                IndexNode shot = new IndexNode("Shot" + Integer.toString(shotcount), data.get(i)[1] / 30.0);
                sencecount++;
                shotcount++;
                scene.addChildren(shot);
                nodes.add(scene);
            } else if (data.get(i)[0] == 2) {
                IndexNode shot = new IndexNode("Shot" + Integer.toString(shotcount), data.get(i)[1] / 30.0);
                nodes.get(sencecount - 2).addChildren(shot);
                shotcount++;
                subshotcount = 1;
            } else if (data.get(i)[0] == 3) {
                IndexNode shot = nodes.get(sencecount - 2).getChildren().get(shotcount - 2);
                if (subshotcount == 1) {
                    IndexNode first = new IndexNode("Subshot" + Integer.toString(subshotcount), shot.getTime());
                    shot.addChildren(first);
                    subshotcount++;
                }
                IndexNode subshot = new IndexNode("Subshot" + Integer.toString(subshotcount), data.get(i)[1] / 30.0);
                shot.addChildren(subshot);
                subshotcount++;

            }
        }
        for (IndexNode node : nodes) {
            System.out.println(node.getName() + " " + node.getTime() * 30);
            for (IndexNode children : node.getChildren()) {
                System.out.println(children.getName() + " " + children.getTime() * 30);
                for (IndexNode subchild : children.getChildren()) {
                    System.out.println(subchild.getName() + " " + subchild.getTime() * 30);
                }
            }
        }

        return nodes;
    }
    public int sequenceAlignment(int[] s1, int[] s2) {
        int gap = 1;
        int mismatch = 1;
        int n = s1.length;
        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            dp[i][0] = i * gap;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j * gap;
        }
        for (int j = 1; j <= n; j++) {
            for (int i = 1; i <= n; i++) {
                if (s1[i - 1] == s2[j - 1])
                    mismatch = 0;
                else
                    mismatch = 1;
                dp[i][j] = Math.min(mismatch + dp[i - 1][j - 1], Math.min(gap + dp[i - 1][j], gap + dp[i][j - 1]));
            }
        }
        return dp[n][n];
    }
    public double calculateHistogramDistance(double[][][][] x, double[][][][] y) {
        double res = 0;
        for (int block = 0; block < 4; block++) {
            for (int r = 0; r < 4; r++) {
                for (int g = 0; g < 4; g++) {
                    for (int b = 0; b < 4; b++) {
                        if (x[block][r][g][b] == 0 && y[block][r][g][b] == 0)
                            continue;
                        double squaredDiff = Math.abs(x[block][r][g][b] - y[block][r][g][b])
                                * Math.abs(x[block][r][g][b] - y[block][r][g][b]);
                        res += squaredDiff / (x[block][r][g][b] + y[block][r][g][b]);
                    }
                }
            }
        }
        return res;
    }
    public void updateCentroids(ArrayList<ArrayList<Integer>> clusters, ArrayList<double[][][][]> centroids, int k) {
        for (int i = 0; i < k; i++) {
            double[][][][] meanHistogram = new double[4][4][4][4];
            for (int block = 0; block < 4; block++) {
                for (int r = 0; r < 4; r++) {
                    for (int g = 0; g < 4; g++) {
                        for (int b = 0; b < 4; b++) {
                            for (int j = 0; j < clusters.get(i).size(); j++) {
                                meanHistogram[block][r][g][b] += histograms.get(keyFrameID
                                        .get(clusters.get(i).get(j)))[block][r][g][b];
                            }
                            meanHistogram[block][r][g][b] = meanHistogram[block][r][g][b] / clusters.get(i).size();
                        }
                    }
                }
            }
            centroids.set(i, meanHistogram);
        }
    }

    public boolean updateClusters(ArrayList<ArrayList<Integer>> clusters, ArrayList<double[][][][]> centroids, int k) {
        boolean res = false;
        ArrayList<ArrayList<Integer>> newClusters = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < k; i++) {
            newClusters.add(new ArrayList<>());
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < clusters.get(i).size(); j++) {
                double minDistance = calculateHistogramDistance(histograms.get(keyFrameID.get(clusters.get(i).get(j))),
                        centroids.get(i));
                int closestC = i;
                for (int c = 0; c < k; c++) {
                    double currDistance = calculateHistogramDistance(histograms.get(keyFrameID.get(clusters.get(i).get(j))),
                            centroids.get(c));
                    if (currDistance < minDistance) {
                        minDistance = currDistance;
                        closestC = c;
                    }
                }
                if (closestC != i)
                    res = true;
                newClusters.get(closestC).add(clusters.get(i).get(j));
            }
        }
        for (int i = 0; i < k; i++) {
            clusters.get(i).clear();
            for (int j = 0; j < newClusters.get(i).size(); j++) {
                clusters.get(i).add(newClusters.get(i).get(j));
            }
        }
        return res;
    }
    public int[] kMeans(ArrayList<Integer> keyFrameID, int k) {
        int n = keyFrameID.size();
        int[] clusterSequence = new int[n];
        ArrayList<double[][][][]> centroids = new ArrayList<>();
        ArrayList<ArrayList<Integer>> clusters = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < k; i++) {
            centroids.add(new double[4][4][4][4]);
            clusters.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            int initialID = i / (n / k);
            if (initialID >= k)
                initialID = k - 1;
            System.out.println(i + " " + initialID);
            clusters.get(initialID).add(i);
        }
        updateCentroids(clusters, centroids, k);
        while (updateClusters(clusters, centroids, k) == true) {
            updateCentroids(clusters, centroids, k);
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < clusters.get(i).size(); j++) {
                    System.out.print(clusters.get(i).get(j) + ", ");
                }
                System.out.print("  ");
            }
            System.out.println();
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < clusters.get(i).size(); j++) {
                clusterSequence[clusters.get(i).get(j)] = i;
            }
        }
        System.out.print("Cluster Sequence: ");
        for (int i = 0; i < n; i++)
            System.out.print(clusterSequence[i] + " ");
        System.out.println();
        return clusterSequence;
    }

    public List<IndexNode> processfile(String rgbUrl) {
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video

        System.out.println(rgbUrl);
        File file = new File(rgbUrl);
        ArrayList<Integer> shots = new ArrayList<Integer>();
        shots.add(0);

        // create an array to store scene change frame ID
        ArrayList<Integer> scenes = new ArrayList<Integer>();
        scenes.add(0);
        ArrayList<Integer[]> data=new ArrayList<>();

        // create an array to store luminance component
        // int[][][] luminances = new int[numFrames][width][height];

        // create an array to indicate candidates


        // read the video file and store each frame
        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            BufferedImage image1 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int flag=0;
            while(true) {
                buffer.clear();
                int eof = channel.read(buffer);
                buffer.rewind();
                if (eof == -1) {
                    break;
                }
                double cummulativeDiff = 0;
                double[][][][] hist=new double[4][4][4][4];
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < height / 2; y++) {
                    for (int x = 0; x < width / 2; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        if (flag == 0) {
                            image1.setRGB(x, y, rgb);

                        } else {
                            int rgb1 = image1.getRGB(x, y);
                            int r1 = (rgb1 >> 16) & 0xFF;
                            int g1 = (rgb1 >> 8) & 0xFF;
                            int b1 = (rgb1) & 0xFF;
                            cummulativeDiff += Math.abs(r - r1) + Math.abs(g - g1)
                                    + Math.abs(b - b1);
                            image1.setRGB(x, y, rgb);
                        }
                        image.setRGB(x, y, rgb);
                        hist[0][r / 64][g / 64][b / 64]++;
                    }
                }
                for (int y = 0; y < height / 2; y++) {
                    for (int x = width / 2; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        if (flag == 0) {
                            image1.setRGB(x, y, rgb);

                        } else {
                            int rgb1 = image1.getRGB(x, y);
                            int r1 = (rgb1 >> 16) & 0xFF;
                            int g1 = (rgb1 >> 8) & 0xFF;
                            int b1 = (rgb1) & 0xFF;
                            cummulativeDiff += Math.abs(r - r1) + Math.abs(g - g1)
                                    + Math.abs(b - b1);
                            image1.setRGB(x, y, rgb);
                        }
                        image.setRGB(x, y, rgb);
                        hist[1][r / 64][g / 64][b / 64]++;
                    }
                }
                for (int y = height / 2; y < height; y++) {
                    for (int x = 0; x < width / 2; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        if (flag == 0) {
                            image1.setRGB(x, y, rgb);

                        } else {
                            int rgb1 = image1.getRGB(x, y);
                            int r1 = (rgb1 >> 16) & 0xFF;
                            int g1 = (rgb1 >> 8) & 0xFF;
                            int b1 = (rgb1) & 0xFF;
                            cummulativeDiff += Math.abs(r - r1) + Math.abs(g - g1)
                                    + Math.abs(b - b1);
                            image1.setRGB(x, y, rgb);
                        }
                        image.setRGB(x, y, rgb);
                        hist[2][r / 64][g / 64][b / 64]++;
                    }
                }
                for (int y = height / 2; y < height; y++) {
                    for (int x = width / 2; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        if (flag == 0) {
                            image1.setRGB(x, y, rgb);


                        } else {
                            int rgb1 = image1.getRGB(x, y);
                            int r1 = (rgb1 >> 16) & 0xFF;
                            int g1 = (rgb1 >> 8) & 0xFF;
                            int b1 = (rgb1) & 0xFF;
                            cummulativeDiff += Math.abs(r - r1) + Math.abs(g - g1)
                                    + Math.abs(b - b1);
                            image1.setRGB(x, y, rgb);
                        }
                        image.setRGB(x, y, rgb);
                        hist[3][r / 64][g / 64][b / 64]++;
                    }
                }
                if (flag> 0) {
                    cummulativeDiff = cummulativeDiff / (double) (width * height);
                    cummulativeDiffs.add(cummulativeDiff);
                }
                if(flag==0)
                    flag=1;
                // frames[i] = image;
                histograms.add(hist);
            }
            numFrames=histograms.size();
            channel.close();
            raf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        int[] candidates = new int[numFrames];

        System.out.println("!");
        /*
         * // extract shots from video using 'Cumulative Pixel-To-Pixel'
         * double[] cummulativeDiffs = new double[numFrames - 1];
         * double[] globalDiffs = new double[numFrames - 1];
         * for (int i = 0; i < numFrames - 1; i++) {
         * BufferedImage image1 = frames[i];
         * BufferedImage image2 = frames[i + 1];
         * double cummulativeDiff = 0;
         * double globalDiff = 0;
         * for (int y = 0; y < height; y++) {
         * for (int x = 0; x < width; x++) {
         * int rgb1 = image1.getRGB(x, y);
         * int rgb2 = image2.getRGB(x, y);
         * int a1 = (rgb1 >> 24) & 0xFF;
         * int r1 = (rgb1 >> 16) & 0xFF;
         * int g1 = (rgb1 >> 8) & 0xFF;
         * int b1 = (rgb1) & 0xFF;
         * int a2 = (rgb2 >> 24) & 0xFF;
         * int r2 = (rgb2 >> 16) & 0xFF;
         * int g2 = (rgb2 >> 8) & 0xFF;
         * int b2 = (rgb2) & 0xFF;
         * cummulativeDiff += Math.abs(a1 - a2) + Math.abs(r1 - r2) + Math.abs(g1 - g2)
         * + Math.abs(b1 - b2);
         * globalDiff += a1 - a2 + r1 - r2 + g1 - g2 + b1 - b2;
         * }
         * }
         * cummulativeDiff = cummulativeDiff / (double) (width * height);
         * globalDiff = Math.abs(globalDiff) / (double) (width * height);
         * System.out.print(i);
         * System.out.print("\t");
         * System.out.println(globalDiff);
         * cummulativeDiffs[i] = cummulativeDiff;
         * globalDiffs[i] = globalDiff;
         * }
         * System.out.println("shots:");
         * for (int i = 0; i < numFrames - 1; i++) {
         * if (cummulativeDiffs[i] > 50) {
         * System.out.print(0.0240756 * (i + 1));
         * System.out.print("\t");
         * System.out.println(cummulativeDiffs[i]);
         * }
         * }
         */

        // preprocess to find boundary segments
        int numSegments = numFrames / 20;
        if (numFrames % 20 == 0)
            numSegments--;
        double[] hisDistances = new double[numSegments];
        for (int i = 0; i < numSegments; i++) {
            hisDistances[i] = calculateHistogramDistance(histograms.get(i*20), histograms.get(i * 20 + 20));
            System.out.println("hisDistance from frame " + (i * 20) + " to " + (i * 20 + 20) + ": " + hisDistances[i]);
        }
        double globalMean = hisDistances[0];
        for (int i = 1; i < numSegments; i++) {
            globalMean = globalMean * i / (double) (i + 1) + hisDistances[i] / (double) (i + 1);
        }
        System.out.println("globalMean: " + globalMean);
        int numUnits = numSegments / 10;
        double[] localMean = new double[numUnits];
        double[] localStd = new double[numUnits];
        for (int i = 0; i < numUnits; i++) {
            for (int j = i * 10; j < i * 10 + 10; j++) {
                localMean[i] += hisDistances[j];
            }
            localMean[i] = localMean[i] / 10;
            System.out.println("LocalMean " + i + ": " + localMean[i]);
            double ss = 0;
            for (int j = i * 10; j < i * 10 + 10; j++) {
                ss += Math.pow((hisDistances[j] - localMean[i]), 2);
            }
            localStd[i] = Math.sqrt(ss / 10);
            System.out.println("localStd " + i + ": " + localStd[i]);
        }
        double[] unitThresholds = new double[numUnits];
        for (int i = 0; i < numUnits; i++) {
            unitThresholds[i] = localMean[i] + 0.6 * globalMean * localStd[i] / localMean[i];
            System.out.println("Unit " + i + ": " + unitThresholds[i]);
        }

        for (int i = 0; i < numUnits; i++) {
            for (int j = i * 10; j < i * 10 + 10; j++) {
                if (hisDistances[j] >= unitThresholds[i])
                    System.out.println("Error1: " + hisDistances[j] + " " + unitThresholds[i]);
                if (j > 0 && hisDistances[j] > 4 * hisDistances[j - 1])
                    System.out.println("Error2: " + hisDistances[j] + " " + hisDistances[j - 1]);
                if (j < numSegments - 1 && hisDistances[j] > 4 * hisDistances[j + 1])
                    System.out.println("Error3: " + hisDistances[j] + " " + hisDistances[j + 1]);
                if (hisDistances[j] > globalMean)
                    System.out.println("Error4: " + hisDistances[j] + " " + globalMean);
                if (hisDistances[j] < unitThresholds[i] && (j > 0 && hisDistances[j] <= 3 * hisDistances[j - 1])
                        && (j < numSegments - 1 && hisDistances[j] <= 3 * hisDistances[j + 1])
                        && hisDistances[j] <= globalMean) {
                    System.out.println("Delete segment: " + (j * 20) + " to " + (j * 20 + 19));
                    for (int k = j * 20; k < j * 20 + 20; k++) {
                        candidates[k] = -1;
                    }
                }
            }
        }

        // extract shots from video using Histogram
        double[] histogramDiffs = new double[numFrames - 1];
        for (int i = 0; i < numFrames - 1; i++) {
            if (candidates[i] == -1)
                continue;
            System.out.print("Frame " + i + " to " + (i + 1));
            System.out.print("\t");
            histogramDiffs[i] = calculateHistogramDistance(histograms.get(i), histograms.get(i+1));
            System.out.println(histogramDiffs[i]);
            System.out.print("\t");
            System.out.println("cumu"+ cummulativeDiffs.get(i));

        }


        //calculate array of thresholds to extract shots
        double[] thresholds = new double[(numFrames - 1) / 1000];
        double globalHisDifMean = histogramDiffs[0];
        int count = 1;
        if (candidates[0] == -1) {
            globalHisDifMean = 0;
            count = 0;
        }
        for (int i = 1; i < numFrames - 1; i++) {
            if (candidates[i] == -1)
                continue;
            globalHisDifMean = globalHisDifMean * count / (double) (count + 1) + histogramDiffs[i] / (double) (count + 1);
            count++;
        }
        System.out.println("globalHisDifMean: " + globalHisDifMean);

        double[] localHisDifMean = new double[(numFrames - 1) / 1000];
        double[] localHisDifStd = new double[(numFrames - 1) / 1000];
        for (int i = 0; i < (numFrames - 1) / 1000; i++) {
            count = 0;
            for (int j = i * 1000; j < i * 1000 + 1000; j++) {
                if (candidates[i] == -1)
                    continue;
                localHisDifMean[i] += histogramDiffs[j];
                count++;
            }
            if (i == (numFrames - 1) / 1000 - 1) {
                for (int j = i * 1000 + 1000; j < numFrames - 1; j++) {
                    if (candidates[i] == -1)
                        continue;
                    localHisDifMean[i] += histogramDiffs[j];
                    count++;
                }
            }
            localHisDifMean[i] = localHisDifMean[i] / count;
            System.out.println("localHisDifMean " + i + ": " + localHisDifMean[i]);
            long ss = 0;
            count = 0;
            for (int j = i * 1000; j < i * 1000 + 1000; j++) {
                if (candidates[i] == -1)
                    continue;
                ss += Math.pow((histogramDiffs[j] - localHisDifMean[i]), 2);
                count++;
            }
            if (i == (numFrames - 1) / 1000 - 1) {
                for (int j = i * 1000 + 1000; j < numFrames - 1; j++) {
                    if (candidates[i] == -1)
                        continue;
                    ss += Math.pow((histogramDiffs[j] - localHisDifMean[i]), 2);
                    count++;
                }
            }
            localHisDifStd[i] = Math.sqrt(ss / count);
            System.out.println("localHisDifStd " + i + ": " + localHisDifStd[i]);

            thresholds[i] = localHisDifMean[i] + globalHisDifMean * localHisDifStd[i] / localHisDifMean[i];
            System.out.println("Threshold: "  + i + ": " + thresholds[i]);
        }

        int prev = 0;
        for (int i = 0; i < numFrames; i++) {
            int thresholdID = (i - 1) / 1000;
            if (i > 0 && (i - 1) / 1000 == (numFrames - 1) / 1000) thresholdID = (numFrames - 1) / 1000 - 1;
            if (i > 0 && candidates[i - 1] != -1 && histogramDiffs[i - 1] >= thresholds[thresholdID]) {
                if (i >= prev + 20) {
                    shots.add(i);
                    prev = i;
                }
            }
        }

        System.out.print("Key Frames: ");
        for (int i = 0; i < shots.size() - 1; i++) {
            keyFrameID.add((shots.get(i) + shots.get(i + 1)) / 2);
            System.out.print(((shots.get(i) + shots.get(i + 1)) / 2) + " ");
        }
        keyFrameID.add((shots.get(shots.size() - 1) + numFrames - 1) / 2);
        System.out.println((shots.get(shots.size() - 1) + numFrames - 1) / 2);

        int[] clusterSequence = kMeans(keyFrameID, Math.min(keyFrameID.size() / 10, 7));

        /*
        int w = 3;
        if (keyFrameID.size() > 25)
            w = keyFrameID.size() / 6;
        if (keyFrameID.size() > 42)
            w = 7;
        int[] seqAlignErrorScore = new int[keyFrameID.size() - 2 * w];
        double maxErrorScore = 0;
        for (int i = 0; i < keyFrameID.size() - 2 * w; i++) {
            seqAlignErrorScore[i] = sequenceAlignment(
                    Arrays.copyOfRange(clusterSequence, i, i + w),
                    Arrays.copyOfRange(clusterSequence, i + w, i + w * 2));
            maxErrorScore = Math.max(maxErrorScore, seqAlignErrorScore[i]);
        }
        for (int i = 0; i < keyFrameID.size() - 2 * w; i++) {
            System.out.print(seqAlignErrorScore[i] + " ");
            if (seqAlignErrorScore[i] > 0.8 * maxErrorScore) {
                scenes.add(shots.get(i + w));

            }
        }
        System.out.println();
        */

        for (int i = 1; i < keyFrameID.size(); i++) {
            if (clusterSequence[i] != clusterSequence[i - 1]) scenes.add(shots.get(i));
        }
        ArrayList<Integer> subshots=new ArrayList<>();
        ArrayList<Double> cumumean=new ArrayList<>();
        for(int i =0;i<shots.size();i++) {
            double cumucumu=0;
            double cumucount=0;
            if(i<shots.size()-1){
                if (shots.get(i) + 22 < shots.get(i + 1)) {
                    for (int j = shots.get(i) + 20; j < shots.get(i + 1) - 3; j++) {
                        cumucumu += cummulativeDiffs.get(j + 1);
                        cumucount++;
                    }
                    cumumean.add(cumucumu/cumucount);
                }
                else{
                    cumumean.add(0.0);
                }

            }
            else if(i==shots.size()-1){
                if (shots.get(i) + 23 < numFrames-1) {
                    for (int j = shots.get(i) + 20; j < numFrames - 3; j++) {
                        cumucumu += cummulativeDiffs.get(j + 1);
                        cumucount++;
                    }
                    cumumean.add(cumucumu/cumucount);
                }
                else {
                    cumumean.add(0.0);
                }
            }



        }
        for(int i=0; i<cumumean.size();i++){
            System.out.println("cumumean" + i+" " +cumumean.get(i));
        }
        int subshotflag=0;
        for(int i =0;i<shots.size();i++){
            if(i<shots.size()-1) {
                if (shots.get(i) + 22 < shots.get(i + 1)){
                    for(int j=shots.get(i)+20;j<shots.get(i+1)-3;j++){
                        double twomean=(cummulativeDiffs.get(j)+cummulativeDiffs.get(j+1))/2;

                        if(subshotflag==0&&twomean>10*cumumean.get(i)){
                            subshots.add(j+1);
                            System.out.println("subshot" + j + " " + twomean);
                            j+=20;
                            subshotflag=1;
                        }
                        if(subshotflag==1&&twomean<cumumean.get(i)/10.0){
                            subshots.add(j+1);
                            System.out.println("subshot" + j + " " + twomean);
                            j+=20;
                            subshotflag=0;
                        }
                    }
                }
            }
            else{
                if(shots.get(i)+23<numFrames-1){
                    for(int j=shots.get(i)+20;j<numFrames-3;j++){
                        double twomean=(cummulativeDiffs.get(j)+cummulativeDiffs.get(j+1))/2;
                        if(subshotflag==0&&twomean>10*cumumean.get(i)){
                            subshots.add(j+1);
                            j+=20;
                            subshotflag=0;
                        }
                        if(subshotflag==1&&twomean<cumumean.get(i)/10.0){
                            subshots.add(j+1);
                            System.out.println("subshot" + j + " " + twomean);
                            j+=20;
                            subshotflag=0;
                        }
                    }
                }
            }
            subshotflag=0;

        }

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            int shotID = 0;
            int sceneID = 0;
            int subshotID=0;
            for (int i = 0; i < numFrames; i++) {
                buffer.clear();
                channel.read(buffer);
                buffer.rewind();
                if (shotID < shots.size() && i == shots.get(shotID)) {
                    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int r = buffer.get() & 0xff;
                            int g = buffer.get() & 0xff;
                            int b = buffer.get() & 0xff;
                            int rgb = (r << 16) | (g << 8) | b;
                            image.setRGB(x, y, rgb);
                        }
                    }
                    if (sceneID < scenes.size() && i == scenes.get(sceneID)) {
                        sceneID++;
                    }
                    File outputfile = new File(i + "scene" + sceneID + ".jpg");
                    ImageIO.write(image, "jpg", outputfile);
                    shotID++;

                }
                if(subshotID< subshots.size()&&i==subshots.get(subshotID)){
                    subshotID++;
                    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int r = buffer.get() & 0xff;
                            int g = buffer.get() & 0xff;
                            int b = buffer.get() & 0xff;
                            int rgb = (r << 16) | (g << 8) | b;
                            image.setRGB(x, y, rgb);
                        }
                    }
                    File outputfile = new File(i + "subshots" + subshotID+ ".jpg");
                    ImageIO.write(image, "jpg", outputfile);

                }
            }
            channel.close();
            raf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i=0 ; i<scenes.size();i++){
            data.add(new Integer[]{1,scenes.get(i)});
            for(int j=0;j<shots.size();j++){
                if(j<shots.size()-1&&shots.get(j)==scenes.get(i)){
                    for(int k=0;k<subshots.size();k++){
                        if(subshots.get(k)>shots.get(j)&&subshots.get(k)<shots.get(j+1)){
                            data.add(new Integer[]{3,subshots.get(k)});
                        }
                    }
                }
                if(shots.get(j)>scenes.get(i)){
                    if(i<scenes.size()-1&&shots.get(j)<scenes.get(i+1)){
                        data.add(new Integer[]{2,shots.get(j)});
                        for(int k=0;k<subshots.size();k++){
                            if(subshots.get(k)>shots.get(j)){
                                if(j<shots.size()-1&&subshots.get(k)<shots.get(j+1)){
                                    data.add(new Integer[]{3,subshots.get(k)});
                                }
                                else if(j==shots.size()-1){
                                    data.add(new Integer[]{3,subshots.get(k)});
                                }
                            }

                        }
                    }
                    else if(i==scenes.size() - 1) {
                        data.add(new Integer[]{2, shots.get(j)});
                        for (int k = 0; k < subshots.size(); k++) {
                            if (subshots.get(k) > shots.get(j)) {
                                if (j < shots.size() - 1 && subshots.get(k) < shots.get(j + 1)) {
                                    data.add(new Integer[]{3, subshots.get(k)});
                                } else if (j == shots.size() - 1) {
                                    data.add(new Integer[]{3, subshots.get(k)});
                                }
                            }

                        }
                    }
                }

            }
        }
        for(int i=0;i<scenes.size();i++){
            System.out.println(scenes.get(i));
        }
        for(int i=0;i<shots.size();i++){
            System.out.println(shots.get(i));
        }
        for(int i=0;i<subshots.size();i++){
            System.out.println(subshots.get(i));
        }
        for(int i=0;i<data.size();i++){
            System.out.println(data.get(i)[0]+" "+data.get(i)[1]);
        }

        return  getMockIndexNodes(data);
    }

}
