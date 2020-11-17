// Implemention based on post:
// https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker {
    constructor(maxDisappeared = 50) {
        // initialize the next unique object ID along with two ordered
        // dictionaries used to keep track of mapping a given object
        // ID to its centroid and number of consecutive frames it has
        // been marked as "disappeared", respectively
        this.nextObjectID = 0;
        this.objects = {};
        this.bboxes = {};
        this.disappeared = {};

        // store the number of maximum consecutive frames a given
        // object is allowed to be marked as "disappeared" until we
        // need to deregister the object from tracking
        this.maxDisappeared = maxDisappeared;
    }

    register(centroid, bbox) {
        // when registering an object we use the next available object
        // ID to store the centroid
        this.objects[this.nextObjectID] = centroid;
        this.bboxes[this.nextObjectID] = bbox;
        this.disappeared[this.nextObjectID] = 0;
        this.nextObjectID += 1;
    }

    deregister(objectID) {
        // to deregister an object ID we delete the object ID from
        // both of our respective dictionaries
        delete this.objects[objectID];
        delete this.bboxes[objectID];
        delete this.disappeared[objectID];
    }

    async dispose() {
        this.nextObjectID = 0;
        await this.asyncForEach(Object.keys(this.objects), async (objectID) => {
            this.deregister(objectID);
        })

        // Object.keys(this.objects).forEach(objectID => {
        //     this.deregister(objectID);
        // });
    }

    update(poses, useAllKP=false) {

        // check to see if the list of input bounding box rectangles
        // is empty
        if (poses.length == 0) {

            // loop over any existing tracked objects and mark them
            // as disappeared
            for (const objectID in Object.keys(this.disappeared)) {
                this.disappeared[objectID] += 1;
                
                // if we have reached a maximum number of consecutive
                // frames where a given object has been marked as
                // missing, deregister it
                if (this.disappeared[objectID] > this.maxDisappeared) {
                    this.deregister(objectID);
                }
            }

            // return early as there are no centroids or tracking info
            // to update
            return {objects: this.objects, bboxes: this.bboxes};
        }
        const centroidsBBoxes = this.getCentroidBBoxes(poses, useAllKP);
        const inputCentroids = centroidsBBoxes.centroids;
        const bboxes = centroidsBBoxes.bboxes;

        // if we are currently not tracking any objects take the input
        // centroids and register each of them
        if (Object.keys(this.objects).length == 0) {

            for (let i = 0; i < inputCentroids.length; i++) {
                const centroid = inputCentroids[i];
                const bbox = bboxes[i];
                this.register(centroid, bbox);
                poses[i].id = i;
            }

            // otherwise, are are currently tracking objects so we need to
            // try to match the input centroids to existing object
            // centroids
        } else {
            // grab the set of object IDs and corresponding centroids
            const objectIDs = Object.keys(this.objects);
            const objectCentroids = Object.values(this.objects);


            // compute the distance between each pair of object
            // centroids and input centroids, respectively -- our
            // goal will be to match an input centroid to an existing
            // object centroid
            const D = this.dists(objectCentroids, inputCentroids);

            // in order to perform this matching we must (1) find the
            // smallest value in each row and then (2) sort the row
            // indexes based on their minimum values so that the row
            // with the smallest value is at the *front* of the index
            // list
            const minors = this.arrMin(D);
            const rows = this.argsort(minors);

            // next, we perform a similar process on the columns by
            // finding the smallest value in each column and then
            // sorting using the previously computed row index list
            let cols = this.argmin(D);
            cols = rows.map((id) => cols[id]);

            // in order to determine if we need to update, register,
            // or deregister an object we need to keep track of which
            // of the rows and column indexes we have already examined
            const usedRows = new Set();
            const usedCols = new Set();

            // loop over the combination of the (row, column) index
            // tuples
            const rowsAndCols = this.zip(rows, cols);
            for (let i = 0; i < rowsAndCols.length; i++) {
                const rc = rowsAndCols[i];

                // if we have already examined either the row or
                // column value before, ignore it
                // val
                const row = rc[0];
                const col = rc[1];

                if (usedRows.has(row) || usedCols.has(col)) {
                    continue;
                }

                // otherwise, grab the object ID for the current row,
                // set its new centroid, and reset the disappeared
                // counter
                const objectID = objectIDs[row];
                this.objects[objectID] = inputCentroids[col];
                this.bboxes[objectID] = bboxes[col];
                this.disappeared[objectID] = 0;

                // indicate that we have examined each of the row and
                // column indexes, respectively
                usedRows.add(row);
                usedCols.add(col);

                // set pose id 
                // poses[row].id = col;
            }

            // compute both the row and column index we have NOT yet
            // examined
            const unusedRows = this.difference(new Set(this.range(D.length)), usedRows);
            const unusedCols = this.difference(new Set(this.range(D[0].length)), usedCols);

            // in the event that the number of object centroids is
            // equal or greater than the number of input centroids
            // we need to check and see if some of these objects have
            // potentially disappeared
            if (D.length >= D[0].length) {

                // loop over the unused row indexes
                for (const row of unusedRows) {
                    // grab the object ID for the corresponding row
                    // index and increment the disappeared counter
                    const objectID = objectIDs[row];
                    this.disappeared[objectID] += 1;

                    // check to see if the number of consecutive
                    // frames the object has been marked "disappeared"
                    // for warrants deregistering the object
                    if (this.disappeared[objectID] > this.maxDisappeared) {
                        this.deregister(objectID);
                    }
                }

            } else {
                // otherwise, if the number of input centroids is greater
                // than the number of existing object centroids we need to
                // register each new input centroid as a trackable object
                unusedCols.forEach(col => {
                    poses[col].id = this.nextObjectID;

                    this.register(inputCentroids[col], bboxes[col]);

                });
            }

            // console.log('rows & cols', rowsAndCols);
            for (let i = 0; i < rowsAndCols.length; i++) {
                const rc = rowsAndCols[i];
                const row = rc[0];
                const col = rc[1];

                // set pose id 
                poses[col].id = row;
            }
        }

        // return the set of trackable objects
        return {objects: this.objects, bboxes: this.bboxes};
    }

    getCentroidBBoxes(arr, useAllKP=false) {
        const centroids = [];
        const bboxes = [];
        arr.forEach(person => {
            const nose = person.pose.keypoints[0].position;
            const leftEye = person.pose.keypoints[1].position;
            const rightEye = person.pose.keypoints[2].position;
            const leftEar = person.pose.keypoints[3].position;
            const rightEar = person.pose.keypoints[4].position;

            const xs = [nose.x, leftEye.x, rightEye.x, leftEar.x, rightEar.x];
            const ys = [nose.y, leftEye.y, rightEye.y, leftEar.y, rightEar.y];

            // bounding box from nose, left eye and right eye
            let bb;
            if (useAllKP) {
                bb = {
                    x0: person.boundingBox.minX,
                    y0: person.boundingBox.minY,
                    x1: person.boundingBox.maxX,
                    y1: person.boundingBox.maxY
                };

            } else {

                bb = {
                    x0: min(xs),
                    y0: min(ys),
                    x1: max(xs),
                    y1: max(ys)
                };
            }

            // centroid
            const cx = (bb.x0 + bb.x1) / 2.0;
            const cy = (bb.y0 + bb.y1) / 2.0;
            const centroid = [cx, cy];

            centroids.push(centroid);
            bboxes.push(bb);
        });
        return {centroids: centroids, bboxes: bboxes};
    }

    dists(objects, inputs) {
        const D = []
        for (let i = 0; i < objects.length; i++) {
            const posObj = objects[i];
            const x0 = posObj[0];
            const y0 = posObj[1];
            let d = []
            for (let j = 0; j < inputs.length; j++) {
                const posIn = inputs[j];
                const x1 = posIn[0];
                const y1 = posIn[1];

                d.push(dist(x0, y0, x1, y1));
            }
            D.push(d);
        }
        return D;
    }

    minAndIndex(arr) {
        let min = arr[0];
        let id = 0;
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (v < min) {
                min = v;
                id = i;
            }
        }
        return [min, id];
    }

    arrMin(arr) {

        const minors = [];
        for (let i = 0; i < arr.length; i++) {
            const minId = this.minAndIndex(arr[i])
            minors.push(arr[i][minId[1]]);
        }
        return minors;
    }

    argmin(arr) {
        const ids = [];
        for (let i = 0; i < arr.length; i++) {
            const minId = this.minAndIndex(arr[i])
            ids.push(minId[1]);
        }
        return ids;
    }

    argsort(arr) {
        const ids = [];
        const values = [];
        for (const e of arr.entries()) {
            ids.push(e[0]);
            values.push(e[1]);
        }
        // console.log(ids, values);

        // Tip from:
        // https://stackoverflow.com/questions/46622486/what-is-the-javascript-equivalent-of-numpy-argsort

        const result = ids
            .map((item, index) => [values[index], item]) // add the value to sort by
            .sort(([count1], [count2]) => count1 - count2) // sort by the value data
            .map(([, item]) => item); // extract the sorted items

        return result;
    }

    zip(arr1, arr2) {
        // Tip from: 
        // https://stackoverflow.com/questions/22015684/how-do-i-zip-two-arrays-in-javascript
        return arr1.map((value, idx) => [value, arr2[idx]]);
    }

    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set
    difference(setA, setB) {
        let _difference = new Set(setA);
        for (let elem of setB) {
            _difference.delete(elem);
        }
        return _difference;
    }

    // https://stackoverflow.com/questions/8273047/javascript-function-similar-to-python-range
    range(n) {
        return [...Array(n).keys()];
    }

    //https://codeburst.io/javascript-async-await-with-foreach-b6ba62bbf404
    async asyncForEach(array, callback) {
        for (let index = 0; index < array.length; index++) {
          await callback(array[index], index, array);
        }
    }

};