import networkx as nx
import matplotlib.pyplot as plt
import unittest
import numpy as np
from numpy.linalg import inv

class TestPhysicalDeviceBase(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def testCircuitInit(self):
        self.assertIsNotNone(Circuit(size=(4,3)))

    def testCircuitInited(self):
        circuit = Circuit(size=(4,3))
        self.assertIsNotNone(circuit.g)
        self.assertEqual(circuit.size, (4,3))
        self.assertEqual(len(circuit.connections()), 0)

    def testCircuitConnected(self):
        circuit = Circuit(size=(4,3))
        circuit.connect(0,1,"R1")
        self.assertIsNotNone(circuit.g)
        self.assertEqual(circuit.size, (4,3))
        self.assertEqual(len(circuit.connections()), 1)

    def testCircuitElectronicsClass(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        self.assertIsNotNone(circuit.g)
        self.assertEqual(len(circuit.connections()), 15)

    def exampleCircuitWithTwoCurrentSources(self):
        circuit = Circuit()
        circuit.connect(11,12,label="")
        circuit.connect(12,13,label="R1", impedance=1)
        circuit.connect(13,14,label="")

        circuit.connect(12,22,label="R1", impedance=1)
        circuit.connect(14,24,label="R1", impedance=1)
        circuit.connect(11,31,label="Vs", voltageSource=12)
        circuit.connect(22,23,label="R1", impedance=1)
        circuit.connect(23,24,label="")
        circuit.connect(31,32,label="")
        circuit.connect(32,33,label="")
        circuit.connect(33,34,label="")
        circuit.connect(22,32,label="R1", impedance=1)
        circuit.connect(23,33,label="R1", impedance=1)
        circuit.connect(23,13,label="I_2", currentSource=2)
        circuit.connect(34,24,label="I_4", currentSource=0)
        
        A1,c1 = circuit.loopVoltageConstraints()
        A2,c2 = circuit.nodeCurrentConstraints()
        # current source I_4 is 4 times the current at edge (22, 32)
        indexSource = circuit.edgeIndex((34,24))
        indexRef = circuit.edgeIndex((22, 32))

        A3,c3 = np.zeros(shape=(1, len(circuit.edgeVector()))), np.array([0])
        A3[0,indexSource] = -1
        A3[0,indexRef] = 4

        indexSource = circuit.edgeIndex((23,13))
        A4,c4 = np.zeros(shape=(1, len(circuit.edgeVector()))), np.array([2])
        A4[0,indexSource] = 1

        A = np.concatenate( (A1,A2,A3,A4) )
        c = np.concatenate( (c1,c2,c3,c4) )

        circuit.manualConstraints = (A,c)
        circuit.removeIsolatedNodes()
        return circuit

    def exampleCircuit2(self):
        circuit = Circuit(size=(2,3))
        circuit.connect(0,1,label="", impedance=0)
        circuit.connect(1,3,label="R4", impedance=4)
        circuit.connect(2,3,label="R4", impedance=4)
        circuit.connect(3,5,label="R2", impedance=2)
        circuit.connect(5,4,label="", impedance=0)
        circuit.connect(4,2,label="Vs", voltageSource=10)
        circuit.connect(2,0,label="", impedance=0)
        return circuit

    def exampleCircuitParallelResistor(self):
        circuit = Circuit(size=(2,3))
        circuit.connect(0,1,label="R2", impedance=2)
        circuit.connect(1,3,label="",)
        circuit.connect(2,3,label="R2", impedance=2)
        circuit.connect(3,5,label="")
        circuit.connect(5,4,label="")
        circuit.connect(4,2,label="Vs", voltageSource=10)
        circuit.connect(2,0,label="")
        return circuit

    def exampleCircuitWithSingleLoopVoltageSource(self):
        circuit = Circuit(size=(2,2))
        circuit.connect(0,1,label="")
        circuit.connect(1,3,label="R4", impedance=5)
        circuit.connect(3,2,label="")
        circuit.connect(2,0,label="Vs", voltageSource=10)
        return circuit

    def exampleCircuitWithSingleLoopCurrentSource(self):
        circuit = Circuit(size=(2,2))
        circuit.connect(0,1,label="")
        circuit.connect(1,3,label="R4", impedance=4)
        circuit.connect(3,2,label="")
        circuit.connect(2,0,label="Is", currentSource=2, impedance=1)
        return circuit

    def exampleCircuitCubeOfResistor(self):
        circuit = Circuit(size=(4,4))
        circuit.connect(0,3,label="R", impedance=1)
        circuit.connect(3,15,label="R", impedance=1)
        circuit.connect(15,12,label="R", impedance=1)
        circuit.connect(12,0,label="R", impedance=1)

        circuit.connect(5,6,label="R", impedance=1)
        circuit.connect(6,10,label="R", impedance=1)
        circuit.connect(10,9,label="R", impedance=1)
        circuit.connect(9,5,label="R", impedance=1)

        circuit.connect(0,5,label="R", impedance=1)
        circuit.connect(3,6,label="R", impedance=1)
        circuit.connect(15,10,label="R", impedance=1)
        circuit.connect(12,9,label="R", impedance=1)

        circuit.connect(10,0,label="Vs", voltageSource=10)
        circuit.removeIsolatedNodes()
        return circuit

    def testCurrentSourcesCount(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        self.assertTrue(circuit.hasCurrentSources)
        sources = circuit.currentSources()
        self.assertEqual(len(sources), 2)

    def testVoltageSourcesCount(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        self.assertTrue(circuit.hasVoltageSources)
        sources = circuit.voltageSources()
        self.assertEqual(len(sources), 1)

    def testLoopsDefault(self):
        circuit = self.exampleCircuitWithSingleLoopVoltageSource()
        loops = circuit.kirchoffLoops()
        self.assertIsNotNone(loops)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), len(circuit.g.edges))

    def testLoopsWithCurrentSource(self):
        circuit = self.exampleCircuitWithSingleLoopCurrentSource()
        loops = circuit.kirchoffLoops()
        self.assertIsNotNone(loops)
        self.assertEqual(len(loops), 0)

    def testLoopsDefaultIsSolvable(self):
        circuit = self.exampleCircuitWithSingleLoopVoltageSource()
        self.assertIsNotNone(circuit.edgeCurrentSolution())

    def testLoopsWithCurrentSourceIsSolvable(self):
        circuit = self.exampleCircuitWithSingleLoopCurrentSource()
        self.assertIsNotNone(circuit.edgeCurrentSolution())

    def testRemoveSources(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        self.assertTrue(circuit.hasCurrentSources)

        circuitAfter = Circuit(size=(2,2))
        circuitAfter.g = circuit.withoutCurrentSources()

        self.assertFalse(circuitAfter.hasCurrentSources)


    def testRemoveCurrentSourcesWhenNone(self):
        circuit = self.exampleCircuitWithSingleLoopVoltageSource()
        self.assertFalse(circuit.hasCurrentSources)

        g = circuit.withoutCurrentSources()

        circuitAfter = Circuit(size=(2,2))
        circuitAfter.g = g

        self.assertFalse(circuit.hasCurrentSources)
        self.assertEqual(circuit.edgeVector(), circuitAfter.edgeVector())

    def testSolution(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        self.assertIsNotNone(circuit.edgeCurrentSolution())

    def testDisplayNoSolution(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        circuit.display()

    def testDisplaySolution(self):
        circuit = self.exampleCircuitWithTwoCurrentSources()
        x = circuit.edgeCurrentSolution()
        circuit.displaySolution()

    def testParallelResistor(self):
        circuit = self.exampleCircuitParallelResistor()
        circuit.displaySolution(layout="grid-size")
        solution = circuit.edgeCurrentSolution()
        srcIdx = circuit.edgeIndex((4,2))
        sources = circuit.voltageSources()
        print("Load resistor is: {0:0.3f} ".format(sources[0][2]/solution[srcIdx]))

    def testResistorCube(self):
        circuit = self.exampleCircuitCubeOfResistor()
        circuit.displaySolution(layout="grid-size")

        solution = circuit.edgeCurrentSolution()
        sources = circuit.voltageSources()
        src = sources[0]
        srcIdx = circuit.edgeIndex((src[0],src[1]))
        print(src[2])
        print(srcIdx )

        print("Load resistor is: {0:0.3f} ".format(src[2]/solution[srcIdx]))
        print(circuit.edgeVector())
        print(solution)

        print(circuit.constraints())
        circuit.printConstraintsLaTeX()

    def testGraphviz(self):
        circuit = self.exampleCircuit2()
        circuit.graphviz("/tmp/test.dot")

    def testGridDecLayout(self):
        circuit = Circuit(size=(10,10))
        for n in range(100):
            circuit.g.add_node(n)
        circuit.display(layout="grid-dec")

class Circuit:
    def __init__(self, size=None):
        self.size = size
        if size is not None:
            self.size = size
        self.g = nx.DiGraph()
        self.manualConstraints = None

    def removeIsolatedNodes(self):
        self.g.remove_nodes_from([n for n in self.g.nodes if self.g.degree(n) == 0])

    def connections(self):
        return self.g.edges

    def connect(self, nodeFrom, nodeTo, label, impedance=None, voltageSource=None, currentSource=None):
        self.g.add_edge(nodeFrom, nodeTo, label=label, impedance=impedance, voltageSource=voltageSource, currentSource=currentSource)

    def withoutCurrentSources(self):
        noSources = self.g.copy()

        for edge in self.g.edges:
            data =  self.g.get_edge_data(edge[0], edge[1])

            if data["currentSource"] is not None:
                noSources.remove_edge(edge[0], edge[1])

        return noSources

    def kirchoffLoops(self):
        noCurrentSources = self.withoutCurrentSources()
        gu = noCurrentSources.to_undirected()
        return nx.cycle_basis(gu)

    def edgeIndex(self, edge):
        indices = []
        for i, e in enumerate(self.edgeVector()):
            if (e[0] == edge[0] and e[1] == edge[1]) or (e[0] == edge[1] and e[1] == edge[0]):
                indices.append(i)

        if len(indices) == 1:
            return indices[0]

        return indices

    def edgeVector(self):
        edges = list(self.g.edges)
        edges.sort()
        return edges 

    def currentVector(self):
        return self.edgeVector()

    @property
    def hasCurrentSources(self):
        return len(self.currentSources()) != 0

    def currentSources(self):
        edges = list(self.g.edges(data="currentSource"))
        edges.sort()

        return [ edge for edge in edges if edge[2] is not None]

    @property
    def hasVoltageSources(self):
        return len(self.currentSources())

    def voltageSources(self):
        edges = list(self.g.edges(data="voltageSource"))
        edges.sort()

        return [ edge for edge in edges if edge[2] is not None]

    def constraints(self):
        if self.manualConstraints is None:
            A,c = self.defaultConstraints()
        else:
            A,c = self.manualConstraints
        return A, c 

    def printConstraintsLaTeX(self):
        A, c = self.constraints()
        w,h = A.shape

        print(r"\left( \begin{matrix}")
        for i in range(w):
            for j in range(h-1):
                print("{0:g}".format(A[i][j]), end=' & ')
            print("{0:g} \\\\".format(A[i][h-1]))
        print(r"\end{matrix} \right)")

        print(r"\left( \begin{matrix}")
        for edge in self.edgeVector():
            print(r"I_{{{0}\rightarrow {1}}} \\".format(edge[0], edge[1]))
        print(r"\end{matrix} \right)")
        print("=")
        print(r"\left( \begin{matrix}")
        for v in c:
            print(r"{0:g} \\".format(v))
        print(r"\end{matrix} \right)")

    def graphviz(self, filePath):
        nx.nx_pydot.write_dot(self.g, filePath)

    def defaultConstraints(self):
        currConstraints, currConstants = self.nodeCurrentConstraints()
        voltConstraints, voltConstants = self.loopVoltageConstraints()
        sourceConstraints, sourceConstants = self.currentSourceConstraints()
        return np.concatenate((currConstraints, voltConstraints, sourceConstraints)),np.concatenate( (currConstants, voltConstants, sourceConstants) )

    def nodeCurrentConstraints(self):
        nNodes = len(self.g.nodes)
        
        edges = self.edgeVector()
        nEdges = len(edges)

        constraints = np.zeros(shape=(nNodes, nEdges))
        constants = np.zeros(shape=(nNodes))
        
        for i, node in enumerate(self.g.nodes):
            for j, current in enumerate(edges):
                data =  self.g.get_edge_data(current[0], current[1])
                if current[0] == node:
                    constraints[i][j] =  -1 

                elif current[1] == node:
                    constraints[i][j] =  +1 
        return constraints, constants

    def currentSourceConstraints(self):
        sources = self.currentSources()
        nSources = len(sources)

        edges = self.edgeVector()
        nEdges = len(edges)

        constraints = np.zeros(shape=(nSources, nEdges))
        constants = np.zeros(shape=(nSources))
        
        for i, source in enumerate(sources):
            j = self.edgeIndex( (source[0], source[1]) )
            constraints[i][j] =  1
            constants[i] = source[2]

        return constraints, constants

    def loopVoltageConstraints(self):
        loops = self.kirchoffLoops()
        nLoops = len(loops)

        edges = self.edgeVector()
        nEdges = len(edges)

        constraints = np.zeros(shape=(nLoops, nEdges))
        constants = np.zeros(shape=(nLoops))
        
        for i, loop in enumerate(loops):
            for j, node in enumerate(loop):
                forwardEdge = (loop[j-1], node)
                reverseEdge = (node, loop[j-1])
                for k, edge in enumerate(edges):
                    if forwardEdge == edge:
                        direction = +1
                    elif reverseEdge == edge:
                        direction = -1
                    else:
                        continue

                    data =  self.g.get_edge_data(*edge)
                    if data["impedance"] is not None:
                        constraints[i][k] = -direction * data["impedance"]
                    elif data["voltageSource"] is not None:
                        constants[i] += -direction * data["voltageSource"]

        return constraints, constants

    def edgeCurrentSolution(self):
        A, c = self.constraints()

        if np.linalg.matrix_rank(A) != len(self.currentVector()):
            raise RuntimeError("The system is not fully solvable: rank is {0} vs {1} unknowns".format(np.linalg.matrix_rank(A),len(self.currentVector()) ))

        x = np.linalg.pinv(A)@c
        return x

    def display(self, layout=None):
        nodePos = {}
        if layout is None:
            if self.size is not None:
                layout = "grid-size"
            else:
                layout = "grid-dec"

        if layout == "grid-size" and self.size is not None:
            w,h = self.size
            for n in self.g.nodes:
                nodePos[n] = (n % w, -(n // w))
        elif layout == "grid-dec":
            for n in self.g.nodes:
                nodePos[n] = (n % 10, -(n//10))

        edgeLabels = { (u,v):self.g.get_edge_data(u,v)["label"] for u,v in self.g.edges()}

        nx.draw_networkx_nodes(
            self.g, nodePos, linewidths=3,
            node_size=200, node_color='gray', alpha=0.9
        )

        nx.draw_networkx_labels(
            self.g, nodePos, alpha=0.9
        )

        nx.draw_networkx_edges(
            self.g, nodePos, edge_color='black', width=1,
            node_size=200, alpha=0.9
        )

        nx.draw_networkx_edge_labels(
            self.g, nodePos,
            font_color='black', edge_labels=edgeLabels, verticalalignment='center',
        )

        nx.draw_networkx_labels(self.g, nodePos)
        plt.axis('off')
        plt.show()

    def displaySolution(self, layout=None):
        x = self.edgeCurrentSolution()

        for i, edge in enumerate(self.edgeVector()):
            self.g.edges[edge]["label"] = self.g.edges[edge]["label"] + " ({0:.1f}A)".format(x[i])

        self.display(layout)

if __name__ == "__main__":
    unittest.main()



