"""CiteRank

Requires
--------
numpy, sklearn, scipy

Classes
-------
CitationGraph :
    A class that can be used to extract an adjacency matrix from the authors
    dataset and to compute some weight matrices for the CitationGraph.

CiteRank :
    A class that can be used to compute the PageRank vector for a given citation
    matrix and compute a score.
"""

import math
from collections.abc import Iterable
from difflib import SequenceMatcher
from timeit import default_timer as timer

import numpy as np
numpy = np
from sklearn.preprocessing import Normalizer
import scipy
from scipy import sparse

class CitationGraph:
    """
    This class is used to extract an adjacency matrix from the authors dataset
    and to compute some weight matrices for the CitationGraph; it also provides
    a function to compute the adjacency matrix of a CollaborationGraph. Any other
    weight matrix beside the offered ones can still be stored in this class and
    can be used to compute the final weighted adjacency matrix of the CitationGraph.
    The computed matrices as stored as scipy.sparse.spmatrix in csr format.

    Attributes
    ----------
    A : scipy.sparse.spmatrix
        the adjacency matrix of the CitationGraph (a_ij=1 if i cited j, 0 otherwise)
    Cit : scipy.sparse.spmatrix
        similar to the adjacency matrix of the CitationGraph, but it contains the
        number of citations instead of ones
    Col : scipy.sparse.spmatrix
        the adjacency matrix of the CollaborationGraph, but it contains the number
        of collaborations instead of ones
    pubs_num : numpy.ndarray
        an array which contains the number of publications of each author

    Methods
    -------
    adjacencyMatrix(authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None
        Extracts the adjacency matrix of the CitationGraph from the dataset.

    citationMatrix(authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None
        Extracts a matrix similar to the adjacency matrix of the CitationGraph
        from the dataset, but it contains the number of citations instead of ones.

    collaborationMatrix(authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None
        Extracts the adjacency matrix of the CollaborationGraph, but it contains
        the number of collaborations instead of ones.

    computePubsNum(authors:Iterable) -> None
        Counts the number of publications of each author.

    concatenateWeightMatrix(matrices:list) -> None
        Adds the weight matrices passed in the parameter to the existing ones.

    getMatrixAt(index:int) -> scipy.sparse.spmatrix
        Returns the weight matrix stored at index "index".

    loadAnyMatrix(*args) -> tuple
        Returns the matrices loaded from files specified in args.

    loadSelfMatrices(**kwargs) -> None
        Loads the default matrices which can be computed by this class.

    setWeightMatrix(M:scipy.sparse.spmatrix, index:int) -> None
        Sets the weight matrix at index "index" as M.

    sum_weight_matrices(beta, filename:str=None) -> scipy.sparse.spmatrix
        Sums the weight matrices multiplied by a balance factor.

    weight_2loops(filename:str=None) -> None
        Computes the weight matrix based on loops of two vertices in the graph.

    weight_citations(filename:str=None) -> None
        Computes the weight matrix based on the number of citations.

    weight_collaborations(filename:str=None) -> None
        Computes the weight matrix based on the number of collaborations between
        authors.

    weight_orgs(authors:Iterable, sigma:float=0.75, filename:str=None) -> None
        Computes the weight matrix based on the similarity between the orgs.
    """
    def __init__(self):
        self.A = None
        self.Col = None
        self.pubs_num = None
        self.Cit = None

        self.__max_norm = Normalizer(norm='max').transform
        self.__Wcol = 0
        self.__Wcit = 1
        self.__Wlp = 2
        self.__Worg = 3
        self.__weight_matrices = np.array([None]*4)
        self.__attr_names = ["A","Col","Cit","Wcol","Wcit","Wlp","Worg"]

    def adjacencyMatrix(self, authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None :
        """
        Extracts the adjacency matrix of the CitationGraph from the dataset.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value
        publications_dict : dict
            a dictionary where each pair should have the id of a publication as
            key and the list of its authors as value
        idMap : dict
            a dictionary where each pair should have an author's id from the
            complete authors dataset as key and his new id from the dataset
            obtained after author name disambiguation task as value
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.A"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an iterable but %s was passed"%str(type(authors)))
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(not isinstance(publications_dict,dict)):
            raise TypeError("publications_dict must be a dict but %s was passed"%str(type(publications_dict)))
        if(not isinstance(idMap,dict)):
            raise TypeError("idMap must be a dict but %s was passed"%str(type(idMap)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.A = sparse.lil_matrix((len(authors),len(authors)))
        i = 0
        start = timer()
        # for each author in the dataset (merged authors, impossible to find an
        # author's id that is also in the list of duplicates of another author)
        for author in authors:
            ind = auth_id_ind[author['id']]
            # for each of his publications
            for publication in author['pubs']:
                # for each cited publication
                for ref_id in publication['references']:
                    co_authors = publications_dict[ref_id] # list of authors of ref_id publication
                    # for each co_author of the cited publication
                    for co_author in co_authors:
                        co_ind = auth_id_ind[idMap[co_author]]
                        self.A[ind, co_ind] = 1
            if(i%100==0):
                print("\rComputed: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(authors),mins,secs))
        self.A = self.A.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.A)

    def citationMatrix(self, authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None :
        """
        Extracts a matrix similar to the adjacency matrix of the CitationGraph
        from the dataset, but it contains the number of citations instead of ones.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value
        publications_dict : dict
            a dictionary where each pair should have the id of a publication as
            key and the list of its authors as value
        idMap : dict
            a dictionary where each pair should have an author's id from the
            complete authors dataset as key and his new id from the dataset
            obtained after author name disambiguation task as value
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Cit"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an iterable but %s was passed"%str(type(authors)))
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(not isinstance(publications_dict,dict)):
            raise TypeError("publications_dict must be a dict but %s was passed"%str(type(publications_dict)))
        if(not isinstance(idMap,dict)):
            raise TypeError("idMap must be a dict but %s was passed"%str(type(idMap)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.Cit = sparse.lil_matrix((len(authors),len(authors)))
        i = 0
        start = timer()
        # for each author in the dataset (merged authors, impossible to find an
        # author's id that is also in the list of duplicates of another author)
        for author in authors:
            ind = auth_id_ind[author['id']]
            # for each of his publications
            for publication in author['pubs']:
                # for each cited publication
                for ref_id in publication['references']:
                    cit_authors = publications_dict[ref_id] # list of authors of ref_id publication
                    # for each co_author of the cited publication
                    for cit_author in cit_authors:
                        cit_ind = auth_id_ind[idMap[cit_author]]
                        self.Cit[ind, cit_ind] += 1
            if(i%100==0):
                print("\rComputed: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(authors),mins,secs))
        self.Cit = self.Cit.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.Cit)

    def collaborationMatrix(self, authors:Iterable, auth_id_ind:dict, publications_dict:dict, idMap:dict, filename:str=None) -> None :
        """
        Extracts the adjacency matrix of the CollaborationGraph, but it contains
        the number of collaborations instead of ones.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value
        publications_dict : dict
            a dictionary where each pair should have the id of a publication as
            key and the list of its authors as value
        idMap : dict
            a dictionary where each pair should have an author's id from the
            complete authors dataset as key and his new id from the dataset
            obtained after author name disambiguation task as value
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Col"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an iterable but %s was passed"%str(type(authors)))
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(not isinstance(publications_dict,dict)):
            raise TypeError("publications_dict must be a dict but %s was passed"%str(type(publications_dict)))
        if(not isinstance(idMap,dict)):
            raise TypeError("idMap must be a dict but %s was passed"%str(type(idMap)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.Col = sparse.lil_matrix((len(authors),len(authors)))
        self.pubs_num = np.zeros(len(authors))
        i = 0
        start = timer()
        # for each author in the dataset (merged authors, impossible to find an
        # author's id that is also in the list of duplicates of another author)
        for author in authors:
            ind = auth_id_ind[author['id']]
            self.pubs_num[i] = len(author['pubs'])
            # for each of his publications
            for publication in author['pubs']:
                co_authors = np.array(publications_dict[publication['id']]) # list of authors of the publication
                co_authors = np.delete(co_authors, np.where(co_authors==author['id'])) # removing the i-th author from the list
                # for each author the i-th author collaborated with
                for co_author in co_authors:
                    co_ind = auth_id_ind[idMap[co_author]]
                    self.Col[ind, co_ind] += 1
            if(i%1000==0):
                print("\rComputed: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(authors),mins,secs))
        self.Col = self.Col.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.Col)

    def computePubsNum(self, authors:Iterable) -> None :
        """
        Counts the number of publications of each author. This operation is also
        done when "collaborationMatrix" is invoked.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author

        Returns
        -------
        None
            the result is stored in the class attribute "self.pubs_num"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an iterable but %s was passed"%str(type(authors)))

        self.pubs_num = np.zeros(len(authors))
        i = 0
        start = timer()
        for author in authors:
            self.pubs_num[i] = len(author['pubs'])
            if(i%1000==0):
                print("\rComputed: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(authors),mins,secs))

    def concatenateWeightMatrix(self, matrices:list) -> None :
        """
        Adds the weight matrices passed in the parameter to the existing ones.
        Each element in the parameter must be an istance of a class that extends
        scipy.sparse.spmatrix (any format is accepted).

        Parameters
        ----------
        matrices : list
            a list of instances of scipy.sparse.spmatrix (any format)

        Returns
        -------
        None
            the matrices are added to the already stored ones in this class
        """
        if(not isinstance(matrices,list)):
            raise TypeError("matrices must be a list but %s was passed"%str(type(matrices)))
        if(len(matrices)==0):
            return
        for m in matrices:
            if(not isinstance(m,sparse.spmatrix)):
                raise TypeError("matrices can only contain instances of scipy.sparse.spmatrix but %s was found"%str(type(m)))

        self.__weight_matrices = np.concatenate((self.__weight_matrices, matrices))

    def getMatrixAt(self, index:int) -> scipy.sparse.spmatrix :
        """
        Returns the weight matrix stored at index "index".

        Parameters
        ----------
        index : int
            the index of the desired matrix

        Returns
        scipy.sparse.spmatrix
            the matrix stored at index "index"
        """
        if(not isinstance(index,int)):
            raise TypeError("index must be an integer number but %s was passed"%str(type(index)))
        if(index<0):
            raise ValueError("index must be greater than or equal to zero but %s was passed"%str(index))
        if(index>=self.__weight_matrices.shape[0]):
            raise ValueError("index must be less than the number of currently stored matrices (%s)"%self.__weight_matrices.shape[0])

        return self.__weight_matrices[index]

    def loadAnyMatrix(self, *args) -> tuple :
        """
        Loads from file any number of matrices and returns them in a tuple. If
        this method is used to load this class matrices, e.g. A, they won't be
        stored in the class attributes.

        Parameters
        ----------
        *args :
            each argument is expected to be a string, which is used as a filename;
            no error is raised directly by this method, but errors can occurr if
            any of the arguments isn't a string or points to not existing file

        Returns
        -------
        tuple
            a tuple which contains the loaded scipy.sparse.spmatrix instances
        """
        if(len(args)==0):
            return ()
        vals = np.array([None]*len(args))
        for i in range(len(args)):
            vals[i] = sparse.load_npz(args[i])
        return tuple(vals)

    def loadSelfMatrices(self, **kwargs) -> None :
        """
        Loads the values of the specified matrices from files. In order to load
        a specific matrix from a file, it's required to pass a kwarg with value
        set to the path of the file which contains the matrix (e.g. A="./A.npz"
        to load the adjacency matrix A).

        Parameters
        ----------
        **kwargs :
            one (or more) from { A,Col,Cit,Wcol,Wcit,Wlp,Worg }; other kwargs
            will be simply ignored without raising any error

        Returns
        -------
        None
            the values loaded with valid kwargs will be stored in the class
            attributes
        """
        if(len(kwargs)==0):
            return
        for attr in kwargs:
            if(attr in self.__attr_names):
                path = kwargs[attr]
                if(attr=="A"):
                    self.A = sparse.load_npz(path)
                elif(attr=="Col"):
                    self.Col = sparse.load_npz(path)
                elif(attr=="Cit"):
                    self.Cit = sparse.load_npz(path)
                elif(attr=="Wcol"):
                    self.__weight_matrices[self.__Wcol] = sparse.load_npz(path)
                elif(attr=="Wcit"):
                    self.__weight_matrices[self.__Wcit] = sparse.load_npz(path)
                elif(attr=="Wlp"):
                    self.__weight_matrices[self.__Wlp] = sparse.load_npz(path)
                else: #attr=="Worg"
                    self.__weight_matrices[self.__Worg] = sparse.load_npz(path)
                print("%s attribute loaded"%attr)

    def setWeightMatrix(self, M:scipy.sparse.spmatrix, index:int) -> None :
        """
        Sets the weight matrix at index "index" as M. If the index is greater
        than the number of already stored matrices, no exception is raised and
        the matrix is stored in the first available position.

        Parameters
        ----------
        M : scipy.sparse.spmatrix
            the weight matrix that must be stored
        index : int
            the index where the matrix should be stored

        Returns
        -------
        None
            the matrix is stored in this class
        """
        if(not isinstance(M,sparse.spmatrix)):
            raise TypeError("M must be a scipy.sparse.spmatrix but %s was passed"%str(type(M)))
        if(not isinstance(index,int)):
            raise TypeError("index must be an integer number but %s was passed"%(str(type(M))))
        if(index<0):
            raise ValueError("index must be a positive number but %s was passed"%str(index))

        if(index>self.__weight_matrices.shape[0]):
            self.concatenateWeightMatrix([M])
            return
        self.__weight_matrices[index] = M

    def sum_weight_matrices(self, beta, filename:str=None) -> scipy.sparse.spmatrix:
        """
        Sums the weight matrices (stored in this class) multiplied by a balance
        factor. Any matrix can be added using methods concatenateWeightMatrix,
        setWeightMatrix or the methods which compute one weight matrix (e.g.
        weight_collaborations)

        Parameters
        ----------
        beta : np.generic or np.ndarray
            a 1d array with balance factors that have to be used; if the matrices
            are not changed using the method setWeightMatrix, the first four
            values are used for the matrices computed by the methods
            weight_collaborations, weight_citations, weight_2loops, weight_orgs
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        -------
        scipy.sparse.spmatrix
            the computed sum matrix
        """
        for i in range(self.__weight_matrices.shape[0]):
            if(self.__weight_matrices[i]==None):
                NotImplementedError("can't compute the sum if one of the matrices is None")
        if(not isinstance(beta,(np.generic,np.ndarray))):
            raise TypeError("beta must be one of (np.generic,np.ndarray) but %s was passed"%str(type(beta)))
        for val in beta:
            if(not isinstance(val,float)):
                raise TypeError("beta can only contain float elements but %s was found"%str(type(val)))
        if(beta.shape[0]!=self.__weight_matrices.shape[0]):
            raise ValueError("there must exist a beta value for each of the %s weight matrices"%str(self.__weight_matrices.shape[0]))
        if(round(beta.sum(),2)!=1.0):
            raise ValueError("the values contained in beta must sum up to 1.0")

        W = sparse.csr_matrix(self.__weight_matrices[0].shape)
        for i in range(self.__weight_matrices.shape[0]):
            W += self.__weight_matrices[i].multiply(beta[i])
        if(filename is not None):
            sparse.save_npz(filename, W)
        return W

    def weight_2loops(self, filename:str=None) -> None :
        """
        Computes the weight matrix based on loops of two vertices in the graph.
        This kind of loops represents the situation of two authors citing each
        other and the aim is to penalize the authors who exchange citations more
        often. This is achieved by computing the attitude of an author to do the
        exchanges (number of authors with which the exchange occurred divided by
        the total number of cited authors) and penalizing him in relation to the
        number of citations to a second author who cited back the first
        ((1 - 2loops_num/tot_cits)/cit_num).

        Parameters
        ----------
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        None
            the matrix is stored internally
        """
        if(self.Cit is None):
            raise NotImplementedError("can't compute weighted matrix if Cit is None")
        if(self.A is None):
            raise NotImplementedError("can't compute weighted matrix if A is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        # find the two vertices loops
        A_loops = self.A.multiply(self.A.T)
        # compute the attitude of each author to do the exchange of citations as
        # the number of authors with which the exchange occurred divided by the
        # total number of cited authors
        attitudes = A_loops.getnnz(axis=1)/np.where(self.A.getnnz(axis=1)==0.0, 1.0, self.A.getnnz(axis=1))
        D = sparse.spdiags(attitudes, 0, *A_loops.shape, format="csr")
        # compute a tolerance value for an author's exchanges as one minus his
        # attitude
        Tol = A_loops - (D * A_loops)
        # compute a penalty value for each citation (for each author) as one
        # divided by the number of citations to an author
        Cit_loops = A_loops.multiply(self.Cit)
        Wlp = sparse.lil_matrix(A_loops.shape)
        for i in range(Cit_loops.shape[0]):
            for j in Cit_loops[i].nonzero()[1]:
                Wlp[i, j] = 1.0 / Cit_loops[i, j]
            if(i%1000==0):
                print("\rComputed: %d/%d"%(i,A_loops.shape[0]), end='',flush=True)
        print("\rComputed: %d/%d"%(i,A_loops.shape[0]))
        # compute the final value multiplying the tolerance measure for the
        # penalty values
        Wlp = Tol.multiply(Wlp) + (self.A - A_loops)
        self.__weight_matrices[self.__Wlp] = Wlp.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.__weight_matrices[self.__Wlp])

    def weight_citations(self, filename:str=None) -> None :
        """
        Computes the weight matrix based on the number of citations. The aim is
        to penalize an author's citation to another if the number of citations to
        this one differs considerably from the average number of citations to all
        the others. This is achieved by computing one minus the ratio of the
        absolute value of the difference between the average number of citations
        and the number of citations to the considered author and the maximum
        number of citations (1 - |avg - cit_num|/max_cit).

        Parameters
        ----------
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        None
            the matrix is stored internally
        """
        if(self.Cit is None):
            raise NotImplementedError("can't compute weighted matrix if Cit is None")
        if(self.A is None):
            raise NotImplementedError("can't compute weighted matrix if A is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        avgs = np.array(self.Cit.sum(axis=1)).flatten()/np.where(self.Cit.getnnz(axis=1)==0.0, 1.0, self.Cit.getnnz(axis=1))
        maxs = np.array(self.Cit.max(axis=1).toarray()).flatten()
        maxs[maxs!=0.0] = 1.0 / maxs[maxs!=0.0]
        D = sparse.spdiags(avgs, 0, *self.Cit.shape, format="csr")
        Wcit = np.absolute((D * self.A) - self.Cit)
        D = sparse.spdiags(maxs, 0, *self.Wcit.shape, format="csr")
        Wcit = self.A - (D * Wcit)
        self.__weight_matrices[self.__Wcit] = Wcit.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.__weight_matrices[self.__Wcit])

    def weight_citations2(self, filename:str=None) -> None:
        """
        Computes the weight matrix based on the number of citations. The aim is
        to penalize an author's citation to another if the number of citations to
        this one differs considerably from the average number of citations to all
        the others. This is achieved by computing the max norm of the matrix which
        elements are computed as one minus the ratio of the number of citations
        and the sum of the number of citations and the average number of citations
        (1 - cit_num/(cit_num + avg)).

        Parameters
        ----------
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        None
            the matrix is stored internally
        """
        if(self.Cit is None):
            raise NotImplementedError("can't compute weighted matrix if Cit is None")
        if(self.A is None):
            raise NotImplementedError("can't compute weighted matrix if A is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        avgs = np.array(self.Cit.sum(axis=1)).flatten()/np.where(self.Cit.getnnz(axis=1)==0.0, 1.0, self.Cit.getnnz(axis=1))
        D = sparse.spdiags(avgs, 0, *self.Cit.shape, format="csr")
        denom = (D * self.A) + self.Cit
        Wcit = self.A - sparse.csr_matrix((np.array(self.Cit[self.Cit!=0]/denom[denom!=0])[0], self.Cit.nonzero()), self.Cit.shape)
        Wcit = self.__max_norm(Wcit)
        self.__weight_matrices[self.__Wcit] = Wcit.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.__weight_matrices[self.__Wcit])

    def weight_collaborations(self, filename:str=None) -> None :
        """
        Computes the weight matrix based on the number of collaborations between
        authors. The aim is to penalize an author's citation to another if the
        two have at least one collaboration. This is achieved by computing one
        minus the ratio of the number of collaborations between the two authors
        and the number of publications of the considered author (1 - col_num/pubs).

        Parameters
        ----------
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        None
            the matrix is stored internally
        """
        if(self.pubs_num is None):
            raise NotImplementedError("can't compute weighted matrix if pubs_num is None")
        if(self.Col is None):
            raise NotImplementedError("can't compute weighted matrix if Col is None")
        if(self.A is None):
            raise NotImplementedError("can't compute weighted matrix if A is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        p = self.pubs_num.copy()
        p[p!=0] = 1.0/p[p!=0]
        D = sparse.spdiags(p, 0, *self.Col.shape, format="csr")
        Wcol = self.A - (D * self.Col)
        # removes the negative numbers which occurr if two authors collaborated
        # but didn't cite each other
        Wcol = Wcol.multiply(self.A)
        self.__weight_matrices[self.__Wcol] = Wcol.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.__weight_matrices[self.__Wcol])

    def weight_orgs(self, authors:Iterable, sigma:float=0.75, filename:str=None) -> None :
        """
        Computes the weight matrix based on the similarity between the orgs. The
        aim is to penalize an author's citation to another if their orgs are
        similar, i.e. they probably work in the same institute or organization.
        This is achieved by computing one minus the similarity value of the two
        orgs multiplied by sigma (1 - sim(o1, o2)*sigma); the similarity value
        is computed as an upper bound of the RatcliffObershelp similarity.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        sigma : float, optional
            a damping factor used to consider properties of the orgs (default is
            0.75)
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        None
            the matrix is stored internally
        """
        if(self.A is None):
            raise NotImplementedError("can't compute weighted matrix if A is None")
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an iterable but %s was passed"%str(type(authors)))
        if(not isinstance(sigma,float)):
            raise TypeError("sigma must be a float but %s was passed"%str(type(sigma)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(sigma<0.0 or sigma>1.0):
            raise ValueError("sigma must be greater than 0.0 and less than 1.0 but %s was passed"%str(sigma))

        orgs_list = [None]*len(authors)
        i = 0
        for author in authors:
            orgs_list[i] = author['org']
            if(i%10000==0):
                print("\rCompleted: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        print("\rCompleted: %d/%d"%(i,len(authors)))
        Worg = sparse.lil_matrix((len(authors), len(authors)))
        start = timer()
        for i in range(self.A.shape[0]):
            nnz_cols = self.A[i].nonzero()[1]
            for j in nnz_cols:
                Worg[i, j] = SequenceMatcher(a=orgs_list[i], b=orgs_list[j]).quick_ratio()
            if(i%100==0):
                print("\rComputed: %d/%d"%(i,self.A.shape[0]),end='',flush=True)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i+1,A.shape[0], mins, secs))
        Worg = self.A - Worg.multiply(sigma)
        self.__weight_matrices[self.__Worg] = Worg.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.__weight_matrices[self.__Worg])

class CiteRank:
    """
    A class that can be used to compute the PageRank vector for a given citation
    matrix and compute a score.

    Methods
    -------
    def minMaxScaler(x, features_range:Iterable=(0,1)) -> numpy.ndarray
        Scales the data in the new features range according to the min-max
        scaling formula.

    pagerank(A:scipy.sparse.spmatrix, alpha:float=0.85, personalization:dict=None, max_iter:int=100, tol:float=1.0e-6) -> numpy.ndarray
        Returns the PageRank of the nodes in the graph.

    sorted_ranks(rank_vectors, ids:Iterable, diffs:bool=False) -> numpy.ndarray
        Returns the position of each author in the sorted rank vectors.
    """

    def minMaxScaler(self, x, features_range:Iterable=(0,1)) -> numpy.ndarray:
        """
        Scales the data in the new features range according to the formula:
        x_std = (x - x.min()) / (x.max() - x.min())
        x_new = x_std * (max - min) + min

        Parameters
        ----------
        x : numpy.generic or numpy.ndarray
            the data that must be scaled
        features_range : Iterable, optional
            the new minimum value and the new maximum value (default is (0,1))

        Returns
        -------
        numpy.ndarray
            the new data scaled in the features range
        """
        if(not isinstance(x,(np.generic,np.ndarray))):
            raise TypeError("x must be a numpy.generic or numpy.ndarray istance but %s was passed"%str(type(x)))
        if(not isinstance(features_range,Iterable)):
            raise TypeError("features_range must be an Iterable but %s was passed"%str(type(features_range)))
        if(len(features_range)!=2):
            raise ValueError("features_range must contain exactly two values: the new minimum and the new maximum")
        if(features_range[0]>=features_range[1]):
            raise ValueError("the value of the new minimum, position 0 in features_range, must be less than the value of the new maximum")

        return ((x - x.min()) / (x.max() - x.min())) * (features_range[1] - features_range[0]) + features_range[0]

    def pagerank(self, A:scipy.sparse.spmatrix, alpha:float=0.85, personalization:dict=None, max_iter:int=100, tol:float=1.0e-6) -> numpy.ndarray:
        """"
        Returns the PageRank of the nodes in the graph. PageRank computes a ranking
        of the nodes in a graph based on the structure of the incoming links; it was
        originally designed as an algorithm to rank web pages.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
          the adjacency matrix of the graph; can also contain weights for edges
          insted of ones and zeros
        alpha : float, optional
          damping parameter for PageRank (default is 0.85)
        personalization: dict, optional
          the "personalization vector" consisting of a dictionary where each pair
          should have the index of a node of the graph (int) as key and its
          personalization value as value. A subset of at least one node is required;
          if not specified, a node personalization value will be zero (default is a
          uniform distribution)
        max_iter : int, optional
          maximum number of iterations in power method eigenvalue solver
        tol : float, optional
          rrror tolerance used to check convergence in power method solver

        Returns
        -------
        numpy.ndarray
           the PageRank vector with the values of the nodes
        """
        if(not isinstance(A,sparse.spmatrix)):
            raise TypeError("A should be a scipy.sparse.spmatrix but %s was passed"%str(type(A)))
        if(not isinstance(alpha,float)):
            raise TypeError("alpha should be a float but %s was passed"%str(type(alpha)))
        if(personalization is not None and not isinstance(personalization,dict)):
            raise TypeError("personalization should be a dict but %s was passed"%str(type(personalization)))
        if(not isinstance(max_iter,int)):
            raise TypeError("max_iter should be an integer number but %s was passed"%str(type(max_iter)))
        if(not isinstance(tol,float)):
            raise TypeError("tol should be a float but %s was passed"%str(type(tol)))
        if(alpha<0.0 or alpha>1.0):
            raise ValueError("alpha should be grater than or equal to zero and less than or equal to one but %s was passed"%str(alpha))
        if(max_iter<=0):
            raise ValueError("max_iter should be grater than zero but %s was passed"%str(max_iter))
        if(tol<0.0):
            raise ValueError("tol must be grater than or equal to zero but %s was passed"%str(tol))

        N = A.shape[0]
        if N == 0:
            return np.array([])
        # P matrix: row-stochastic matrix
        S = np.array(A.sum(axis=1)).flatten()
        S[S!=0] = 1.0/S[S!=0]
        D = sparse.spdiags(S, 0, *A.shape, format="csr")
        P = D * A
        # Personalization vector
        if(personalization is None):
            v = np.repeat(1.0/N, N)
        else:
            v = np.array([personalization.get(n, 0) for n in range(N)], dtype=float)
            v = v / v.sum()
        # Dangling nodes
        dangling_weights = np.repeat(1.0/N, N)
        is_dangling = np.where(np.array(P.sum(axis=1)).flatten()==0.0)[0]
        # initial vector
        r = np.repeat(1.0/N, N)
        # power iteration: make up to max_iter iterations
        convergence = False
        for i in range(max_iter):
            rlast = r
            r = alpha * (r * P + sum(r[is_dangling]) * dangling_weights) + (1 - alpha) * v
            # check convergence, l1 norm
            err = np.absolute(r - rlast).sum()
            if(err < N * tol):
                convergence = True
                break
            print("\rIterations: %d/%d"%(i+1,max_iter),end='',flush=True)
        print("\rIterations: %d/%d"%(i+1,max_iter))
        if(not convergence):
            raise RuntimeError("power iteration failed to converge within %d iterations"%max_iter)
        else:
            return r

    def sorted_ranks(self, rank_vectors, ids:Iterable, diffs:bool=False) -> numpy.ndarray:
        """
        Returns the position of each author in the sorted rank vectors.

        Parameters
        ----------
        rank_vectors : numpy.generic or numpy.ndarray
            the rank vector(s) from which the positions are computed
        ids : Iterable
            the ids of all the authors
        diffs : bool, optional
            if True, the difference between the position vectors is computed and
            returned (default is False)

        Returns
        -------
        numpy.ndarray
            the positions vector or a tuple containing the positions vectors and
            the difference between them
        """
        if(not isinstance(rank_vectors,(np.generic,np.ndarray))):
            raise TypeError("rank_vectors must be a numpy.ndarray or numpy.generic but %s was passed"%str(type(rank_vectors)))
        if(not isinstance(ids,Iterable)):
            raise TypeError("ids must be an Iterable but %s was passed"%str(type(ids)))
        if(not isinstance(diffs,type(True))):
            raise TypeError("diffs must be a boolean but %s was passed"%str(type(diffs)))
        if(rank_vectors.shape[0]<1 or rank_vectors.shape[0]>2):
            raise ValueError("rank_vectors shape on axis=0 must be equal to 1 or 2, but %s was passed"%rank_vectors.shape[0])
        if(rank_vectors.shape[0]<2 and diffs):
            raise ValueError("rank_vectors shape on axis=0 must be equal to 2 when diffs=True")

        ranked_ids = ids[np.argsort(-rank_vectors)]
        dicts = np.array([dict(zip(ranked_ids[i], np.arange(1, ranked_ids.shape[1]+1, 1))) for i in range(ranked_ids.shape[0])])
        ranks = np.array([np.array([dicts[i][idx] for idx in ids]) for i in range(ranked_ids.shape[0])])
        return ranks if rank_vectors.shape[0]==1 else tuple(np.vstack((ranks, np.diff(np.flip(ranks, axis=0), axis=0)[0])))
