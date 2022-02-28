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

    def testCircuitClaudine(self):
        circuit, constraints = self.exampleCircuit1()
        self.assertIsNotNone(circuit.g)
        self.assertEqual(circuit.size, (4,3))
        self.assertEqual(len(circuit.connections()), 15)

    def exampleCircuit1(self):
        circuit = Circuit(size=(4,3))
        circuit.connect(0,1,label="")
        circuit.connect(1,2,label="R1", impedance=1)
        circuit.connect(1,5,label="R1", impedance=1)
        circuit.connect(2,3,label="")
        circuit.connect(3,7,label="R1", impedance=1)
        circuit.connect(8,0,label="Vs", voltageSource=12)
        circuit.connect(5,6,label="R1", impedance=1)
        circuit.connect(6,7,label="")
        circuit.connect(9,8,label="")
        circuit.connect(9,10,label="")
        circuit.connect(10,11,label="")
        circuit.connect(5,9,label="R1", impedance=1)
        circuit.connect(6,10,label="R1", impedance=1)
        circuit.connect(6,2,label="I_2", currentSource=2)
        circuit.connect(11,7,label="I_4", currentSource=0)

        A1,c1 = circuit.loopVoltageConstraints()
        A2,c2= circuit.nodeCurrentConstraints()
        A3,c3 = np.array([[0,0,0,0,0,0,4,0,0,0,0,0,0,0,-1]]), np.array([0]) 
        A4,c4 = np.array([[0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0]]), np.array([2]) 
        A = np.concatenate( (A1,A2,A3,A4) )
        c = np.concatenate( (c1,c2,c3,c4) )

        return circuit, (A,c)

    def exampleCircuit2(self):
        circuit = Circuit(size=(2,3))
        circuit.connect(0,1,label="", impedance=0)
        circuit.connect(1,3,label="R4", impedance=4)
        circuit.connect(2,3,label="R4", impedance=4)
        circuit.connect(3,5,label="R2", impedance=2)
        circuit.connect(5,4,label="", impedance=0)
        circuit.connect(4,2,label="Vs", voltageSource=10)
        circuit.connect(2,0,label="", impedance=0)
        return circuit, None

    def exampleCircuit3(self):
        circuit = Circuit(size=(2,3))
        circuit.connect(0,1,label="", impedance=0)
        circuit.connect(1,3,label="R4", impedance=4)
        circuit.connect(2,3,label="R4", impedance=4)
        circuit.connect(3,5,label="R2", impedance=2)
        circuit.connect(5,4,label="", impedance=0)
        circuit.connect(4,2,label="Is", currentSource=10)
        circuit.connect(2,0,label="", impedance=0)
        return circuit, None

    def exampleCircuitWithSingleLoopVoltageSource(self):
        circuit = Circuit(size=(2,2))
        circuit.connect(0,1,label="")
        circuit.connect(1,3,label="R4", impedance=5)
        circuit.connect(3,2,label="")
        circuit.connect(2,0,label="Vs", voltageSource=10)
        return circuit, None

    def exampleCircuitWithSingleLoopCurrentSource(self):
        circuit = Circuit(size=(2,2))
        circuit.connect(0,1,label="")
        circuit.connect(1,3,label="R4", impedance=4)
        circuit.connect(3,2,label="")
        circuit.connect(2,0,label="Is", currentSource=2, impedance=1)
        return circuit, None

    def testLoopsDefault(self):
        circuit, constraints = self.exampleCircuitWithSingleLoopVoltageSource()
        loops = circuit.kirchoffLoops()
        self.assertIsNotNone(loops)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), len(circuit.g.edges))

    def testLoopsWithCurrentSource(self):
        circuit, constraints = self.exampleCircuitWithSingleLoopCurrentSource()
        loops = circuit.kirchoffLoops()
        self.assertIsNotNone(loops)
        self.assertEqual(len(loops), 0)

    def testLoopsDefaultIsSolvable(self):
        circuit, constraints = self.exampleCircuitWithSingleLoopVoltageSource()
        self.assertIsNotNone(circuit.edgeCurrentSolution())

    def testLoopsWithCurrentSourceIsSolvable(self):
        circuit, constraints = self.exampleCircuitWithSingleLoopCurrentSource()
        self.assertIsNotNone(circuit.edgeCurrentSolution())

    # def testNodes(self):
    #     circuit, constraints = self.exampleCircuit2()
    #     self.assertIsNotNone(circuit.kirchoffNodes())        

    # def testNodesConstraints(self):
    #     circuit, constraints = self.exampleCircuit3()

    # def testLoopsConstraints(self):
    #     circuit, constraints = self.exampleCircuit2()
    #     print(circuit.loopVoltageConstraints())        

    @unittest.skip("Display")
    def testConstraints(self):
        circuit, constraints = self.exampleCircuit2()
        A,c = circuit.constraints()
        
        print()
        print(circuit.currentVector())
        print(np.linalg.pinv(A)@c)
        circuit.display()

    # def testCurrentSources(self):
    #     circuit = self.exampleCircuit3()
    #     print(circuit.currentSources())
    #     print(circuit.currentSourceConstraints())

    @unittest.skip("Display")
    def testDisplay(self):
        circuit, constraints = self.exampleCircuit2()
        circuit.display()

    def testNoSources(self):
        print("ICI")
        circuit, constraints = self.exampleCircuit3()
        print(circuit.g.edges)
        no = circuit.withoutCurrentSources()
        no.display()

    def testNoSources(self):
        circuit, constraints = self.exampleCircuit1()
        circuit.g.remove_node(4)
        print(circuit.currentVector())
        A1,c1 = circuit.loopVoltageConstraints()
        # print(A1,c1)
        A2,c2= circuit.nodeCurrentConstraints()
        # print(circuit.currentSourceConstraints())

        A3,c3 = np.array([[0,0,0,0,0,0,4,0,0,0,0,0,0,0,-1]]), np.array([0]) 
        A4,c4 = np.array([[0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0]]), np.array([2]) 
        A = np.concatenate( (A1,A2,A3,A4) )
        c = np.concatenate( (c1,c2,c3,c4) )
        
        print(A)
        print(c)
        self.assertEqual(np.linalg.matrix_rank(A), len(circuit.currentVector()))
        print()
        # print()
        print(circuit.currentVector())
        x = np.linalg.pinv(A)@c
        print(x)
        for i, edge in enumerate(circuit.currentVector()):
            circuit.g.edges[edge]["label"] += " ({0:.2f}A)".format(x[i])
        circuit.display()


class Circuit:
    def __init__(self, size):
        self.size = size
        self.g = nx.DiGraph()
        self.manualConstraints = None
        for i in range(size[0]):
            for j in range(size[1]):
                self.g.add_node(size[0]*j+i)

    def withoutCurrentSources(self):
        noSources = self.g.copy()

        for edge in self.g.edges:
            data =  self.g.get_edge_data(edge[0], edge[1])

            if data["currentSource"] is not None:
                noSources.remove_edge(edge[0], edge[1])

        return noSources

    def connections(self):
        return self.g.edges

    def connect(self, nodeFrom, nodeTo, label, impedance=None, voltageSource=None, currentSource=None):
        self.g.add_edge(nodeFrom, nodeTo, label=label, impedance=impedance, voltageSource=voltageSource, currentSource=currentSource)

    def kirchoffLoops(self):
        noCurrentSources = self.withoutCurrentSources()
        gu = noCurrentSources.to_undirected()
        return nx.cycle_basis(gu)

    def kirchoffNodes(self):
        nodes = {}
        for n in self.g.nodes: 
            nodes[n] = {"out":list(self.g.out_edges(n)), "in":list(self.g.in_edges(n))}
        return nodes

    def edgeVector(self):
        edges = list(self.g.edges)
        edges.sort()
        return edges 

    def currentVector(self):
        return self.edgeVector()

    def currentSources(self):
        edges = list(self.g.edges(data="currentSource"))
        edges.sort()

        return [ edge for edge in edges if edge[2] is not None]

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

    def defaultConstraints(self):
        currConstraints, currConstants = self.nodeCurrentConstraints()
        voltConstraints, voltConstants = self.loopVoltageConstraints()
        sourceConstraints, sourceConstants = self.currentSourceConstraints()
        return np.concatenate((currConstraints, voltConstraints, sourceConstraints)),np.concatenate( (currConstants, voltConstants, sourceConstants) )

    def nodeCurrentConstraints(self):
        nNodes = len(self.g.nodes)
        
        currents = self.currentVector()
        nCurrents = len(currents)

        constraints = np.zeros(shape=(nNodes, nCurrents))
        constants = np.zeros(shape=(nNodes))
        
        for i, node in enumerate(self.g.nodes):
            for j, current in enumerate(currents):
                data =  self.g.get_edge_data(current[0], current[1])
                if current[0] == node:
                    constraints[i][j] =  -1 

                elif current[1] == node:
                    constraints[i][j] =  +1 
        return constraints, constants

    def currentSourceConstraints(self):
        currents = self.currentVector()
        nCurrents = len(currents)

        sources = self.currentSources()
        nSources = len(sources)

        constraints = np.zeros(shape=(nSources, nCurrents))
        constants = np.zeros(shape=(nSources))
        
        for i, source in enumerate(sources):
            for j, current in enumerate(currents):
                data =  self.g.get_edge_data(source[0], source[1])
                if current[0] == source[0] and current[1] == source[1]:
                    constraints[i][j] =  1
                    constants[i] = source[2]
                elif current[0] == source[1] and current[1] == source[0]:
                    constraints[i][j] =  -1
                    constants[i] = source[2]
                    
        return constraints, constants

    def loopVoltageConstraints(self):
        loops = self.kirchoffLoops()
        nLoops = len(loops)

        currents = self.currentVector()
        nCurrents = len(currents)

        constraints = np.zeros(shape=(nLoops, nCurrents))
        constants = np.zeros(shape=(nLoops))
        
        for i, loop in enumerate(loops):
            for j, node in enumerate(loop):
                forwardEdge = (loop[j-1], node)
                reverseEdge = (node, loop[j-1])
                for k, current in enumerate(currents):
                    if forwardEdge == current:
                        data =  self.g.get_edge_data(forwardEdge[0], forwardEdge[1])
                        if data["impedance"] is not None:
                            constraints[i][k] =  -1 * data["impedance"]
                        elif data["voltageSource"] is not None:
                            constants[i] -= data["voltageSource"]
                    elif reverseEdge == current:
                        data =  self.g.get_edge_data(reverseEdge[0], reverseEdge[1])
                        if data["impedance"] is not None:
                            constraints[i][k] = +1 * data["impedance"]
                        elif data["voltageSource"] is not None:
                            constants[i] += data["voltageSource"]

        return constraints, constants

    def isSolvable(self):
        if self.constraints is None:
            A,c = self.defaultConstraints()
        else:
            A,c = self.constraints

        if np.linalg.matrix_rank(A) != len(self.currentVector()):
            raise False

        return True

    def edgeCurrentSolution(self):
        A, c = self.constraints()

        if np.linalg.matrix_rank(A) != len(self.currentVector()):
            raise RuntimeError("The system is not fully solvable: rank is {0} vs {1} unknowns".format(np.linalg.matrix_rank(A),len(self.currentVector()) ))

        x = np.linalg.pinv(A)@c
        return x

    def display(self):
        w,h = self.size
        nodePos = {}
        for n in self.g.nodes:
            nodePos[n] = (n%w, -(n//w))

        edgeLabels = { (u,v):self.g.get_edge_data(u,v)["label"] for u,v in self.g.edges()}

        nx.draw(
            self.g, nodePos, edge_color='black', width=3, linewidths=3,
            node_size=100, node_color='pink', alpha=0.9
        )

        nx.draw_networkx_edge_labels(
            self.g, nodePos,
            font_color='red', edge_labels=edgeLabels
        )

        nx.draw_networkx_labels(self.g, nodePos)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    unittest.main()



