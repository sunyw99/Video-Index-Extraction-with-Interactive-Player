package view;

import model.IndexNode;
import tool.ProcessTool;
import uk.co.caprica.vlcj.player.base.ControlsApi;
import uk.co.caprica.vlcj.player.base.StatusApi;
import uk.co.caprica.vlcj.player.component.EmbeddedMediaPlayerComponent;

import javax.swing.*;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class VideoPlayer extends JPanel {
    EmbeddedMediaPlayerComponent mediaPlayerComponent;

    JPanel buttonBar;
    JButton playButton;
    JButton pauseButton;
    JButton stopButton;

    List<Long> timeBoundaries = new ArrayList<>();
    List<DefaultMutableTreeNode> timeBoundaryNodes = new ArrayList<>();
    Set<Double> timeSet = new HashSet<>();

    JPanel indexPanel;
    JTree tree;

    public void init(List<IndexNode> nodes, String videoUrl) {

        setLayout(new BorderLayout());

        initMedia();

        initButtons();

        initTree(nodes);

        initLoopWorker();

        mediaPlayerComponent.mediaPlayer().media().play(videoUrl);
    }

    private void initMedia() {
        mediaPlayerComponent = new EmbeddedMediaPlayerComponent();
        add(mediaPlayerComponent, BorderLayout.CENTER);
    }

    private void initButtons() {
        buttonBar = new JPanel();
        playButton = new JButton("Play");
        pauseButton = new JButton("Pause");
        stopButton = new JButton("Stop");

        buttonBar.add(playButton);
        playButton.addActionListener(e -> playVideo());
        buttonBar.add(pauseButton);
        pauseButton.addActionListener(e -> pauseVideo());
        buttonBar.add(stopButton);
        stopButton.addActionListener(e -> stopVideo());
        add(buttonBar, BorderLayout.SOUTH);
    }

    private void initTree(List<IndexNode> nodes) {
        timeBoundaries = new ArrayList<>();
        timeSet = new HashSet<>();

        DefaultMutableTreeNode rootNode = new DefaultMutableTreeNode(new IndexNode("Root", 0));

        for (IndexNode scene : nodes) {
            DefaultMutableTreeNode sceneNode = new DefaultMutableTreeNode(scene);
            addTimePoint(scene.getTime(), scene, sceneNode);
            rootNode.add(sceneNode);
            if (!scene.isLeaf()) {
                for (int j = 0; j < scene.getChildren().size(); j++) {
                    IndexNode shot = scene.getChildren().get(j);
                    DefaultMutableTreeNode shotNode = new DefaultMutableTreeNode(shot);
                    addTimePoint(shot.getTime(), shot, shotNode);
                    sceneNode.add(shotNode);
                    if (!shot.isLeaf()) {
                        for (int k = 0; k < shot.getChildren().size(); k++) {
                            IndexNode subshot = shot.getChildren().get(k);
                            DefaultMutableTreeNode subshotNode = new DefaultMutableTreeNode(subshot);
                            addTimePoint(subshot.getTime(), subshot, subshotNode);
                            shotNode.add(subshotNode);
                        }
                    }
                }
            }
        }

        tree = new JTree(rootNode);
        tree.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                TreePath path = tree.getPathForLocation(e.getX(), e.getY());
                if (path != null) {
                    DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
                    if (node != null && node.isLeaf()) {
                        IndexNode nodeData = (IndexNode) node.getUserObject();
                        System.out.println("set time is " + nodeData.getTime());
                        playVideo(nodeData.getTime() + 0.1);
                    }
                }
            }
        });
        JScrollPane scrollPane = new JScrollPane(tree);
        scrollPane.setPreferredSize(new Dimension(150, 500));
        indexPanel = new JPanel();
        indexPanel.add(scrollPane);

        add(indexPanel, BorderLayout.WEST);
    }

    private void initLoopWorker() {
        LoopWorker loopWorker = new LoopWorker();
        loopWorker.execute();
    }

    private void addTimePoint(double timePoint, IndexNode node, DefaultMutableTreeNode treeNode) {
        if (!timeSet.contains(timePoint) && node.isLeaf()) {
            timeSet.add(timePoint);
            timeBoundaries.add((long)(timePoint * 1000));
            timeBoundaryNodes.add(treeNode);
        }
    }

    private double findStopTime() {
        double curTime = mediaPlayerComponent.mediaPlayer().status().time();
        double prevTimeBoundary = 0;
        for (double timeBoundary : timeBoundaries) {
            if (timeBoundary > curTime) {
                return prevTimeBoundary;
            }
            prevTimeBoundary = timeBoundary;
        }
        return timeBoundaries.get(timeBoundaries.size() - 1);
    }


    private void pauseVideo() {
        if (mediaPlayerComponent.mediaPlayer().status().isPlaying()) {
            mediaPlayerComponent.mediaPlayer().controls().pause();
        }
    }

    private void playVideo() {
        mediaPlayerComponent.mediaPlayer().controls().play();
    }

    private void playVideo(double second) {
        System.out.println((long) (second * 1000));
        mediaPlayerComponent.mediaPlayer().controls().setTime((long) (second * 1000));
    }

    private void stopVideo() {
        mediaPlayerComponent.mediaPlayer().controls().setTime((long) (findStopTime()));
        if (mediaPlayerComponent.mediaPlayer().status().isPlaying()) {
            mediaPlayerComponent.mediaPlayer().controls().pause();
        }
    }

    class LoopWorker extends SwingWorker<Void, Void> {
        @Override
        protected Void doInBackground() throws Exception {
            while (true) {
                if (mediaPlayerComponent.mediaPlayer().status().isPlaying()) {
                    DefaultMutableTreeNode curNode = timeBoundaryNodes.get(timeBoundaryNodes.size() - 1);
                    long curTime = mediaPlayerComponent.mediaPlayer().status().time();
                    for (int i = 1; i < timeBoundaries.size(); i++) {
                        if (timeBoundaries.get(i) > curTime) {
                            curNode = timeBoundaryNodes.get(i - 1);
                            //System.out.println(curTime);
                            //System.out.println(i);
                            //System.out.println(timeBoundaries.get(i - 1));
                            //System.out.println("-----------");
                            TreePath pathToNode = new TreePath(curNode.getPath());
                            tree.setSelectionPath(pathToNode);
                            break;
                        }
                    }
                }

                try {
                    // Add a delay to prevent excessive resource usage
                    Thread.sleep(600);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
