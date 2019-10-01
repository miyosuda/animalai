using System.Linq;
using System;
using UnityEngine;
using Random = UnityEngine.Random;
using MLAgents;
using PrefabInterface;
using System.Collections.Generic;

public class TrainingAgent : Agent, IPrefab
{
    public void RandomSize() { }
    public void SetColor(Vector3 color) { }
    public void SetSize(Vector3 scale) { }

    public virtual Vector3 GetPosition(Vector3 position,
                                        Vector3 boundingBox,
                                        float rangeX,
                                        float rangeZ)
    {
        float xBound = boundingBox.x;
        float zBound = boundingBox.z;
        float xOut = position.x < 0 ? Random.Range(xBound, rangeX - xBound)
                                    : Math.Max(0, Math.Min(position.x, rangeX));
        float yOut = Math.Max(position.y, 0) + transform.localScale.y / 2 + 0.01f;
        float zOut = position.z < 0 ? Random.Range(zBound, rangeZ - zBound)
                                    : Math.Max(0, Math.Min(position.z, rangeZ));

        return new Vector3(xOut, yOut, zOut);
    }

    public virtual Vector3 GetRotation(float rotationY)
    {
        return new Vector3(0,
                        rotationY < 0 ? Random.Range(0f, 360f) : rotationY,
                        0);
    }

    public float speed = 30f;
    public float rotationSpeed = 100f;
    public float rotationAngle = 0.25f;
    [HideInInspector]
    public int numberOfGoalsCollected = 0;

    private Rigidbody _rigidBody;
    private bool _isGrounded;
    private ContactPoint _lastContactPoint;
    private TrainingArea _area;
    private float _rewardPerStep;
    private Color[] _allBlackImage;
    private PlayerControls _playerScript;

    private LIDAR _lidar = new LIDAR();


    public override void InitializeAgent()
    {
        _area = GetComponentInParent<TrainingArea>();
        _rigidBody = GetComponent<Rigidbody>();
        _rewardPerStep = agentParameters.maxStep > 0 ? -1f / agentParameters.maxStep : 0;
        _playerScript = GameObject.FindObjectOfType<PlayerControls>();
    }

    public override void CollectObservations()
    {
        Vector3 localVel = transform.InverseTransformDirection(_rigidBody.velocity);
        AddVectorObs(localVel);

        // CHANGED: Custumly added global positaion and rotation
        Vector3 position = _rigidBody.position;
        AddVectorObs(position);

        float rotation = transform.eulerAngles.y;
        AddVectorObs(rotation);

        // Lidarでスキャンした結果をvector observationに追加
        _lidar.Process(transform);
        AddVectorObs(_lidar.Result);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        int actionForward = Mathf.FloorToInt(vectorAction[0]);
        int actionRotate = Mathf.FloorToInt(vectorAction[1]);

        moveAgent(actionForward, actionRotate);

        AddReward(_rewardPerStep);
    }

    private void moveAgent(int actionForward, int actionRotate)
    {
        Vector3 directionToGo = Vector3.zero;
        Vector3 rotateDirection = Vector3.zero;

        if (_isGrounded)
        {
            switch (actionForward)
            {
                case 1:
                    directionToGo = transform.forward * 1f;
                    break;
                case 2:
                    directionToGo = transform.forward * -1f;
                    break;
            }
        }
        switch (actionRotate)
        {
            case 1:
                rotateDirection = transform.up * 1f;
                break;
            case 2:
                rotateDirection = transform.up * -1f;
                break;
        }

        transform.Rotate(rotateDirection, Time.fixedDeltaTime * rotationSpeed);
        _rigidBody.AddForce(directionToGo * speed * Time.fixedDeltaTime, ForceMode.VelocityChange);
    }

    public override void AgentReset()
    {
        _playerScript.prevScore = GetCumulativeReward();
        numberOfGoalsCollected = 0;
        _area.ResetArea();
        _rewardPerStep = agentParameters.maxStep > 0 ? -1f / agentParameters.maxStep : 0;
        _isGrounded = false;
    }


    void OnCollisionEnter(Collision collision)
    {
        foreach (ContactPoint contact in collision.contacts)
        {
            if (contact.normal.y > 0)
            {
                _isGrounded = true;
            }
        }
        _lastContactPoint = collision.contacts.Last();
    }

    void OnCollisionStay(Collision collision)
    {
        foreach (ContactPoint contact in collision.contacts)
        {
            if (contact.normal.y > 0)
            {
                _isGrounded = true;
            }
        }
        _lastContactPoint = collision.contacts.Last();
    }

    void OnCollisionExit(Collision collision)
    {
        if (_lastContactPoint.normal.y > 0)
        {
            _isGrounded = false;
        }
    }

    public void AgentDeath(float reward)
    {
        AddReward(reward);
        Done();
    }

    public void AddExtraReward(float rewardFactor)
    {
        AddReward(Math.Min(rewardFactor * _rewardPerStep,-0.00001f));
    }

    public override bool LightStatus()
    {
        return _area.UpdateLigthStatus(GetStepCount());
    }

    public class LIDAR 
    {
        private string[] targetNames = {
            "CylinderTunnel(Clone)",
            "CylinderTunnelTransparent(Clone)",
            "Ramp(Clone)",
            "Wall(Clone)",
            "WallTransparent(Clone)",
            "Cardbox1(Clone)",
            "Cardbox2(Clone)",
            "LObject(Clone)",
            "LObject2(Clone)",
            "UObject(Clone)",
            "BadGoal(Clone)",
            "BadGoalBounce(Clone)",
            "DeathZone(Clone)",
            "GoodGoal(Clone)",
            "GoodGoalBounce(Clone)",
            "GoodGoalMulti(Clone)",
            "GoodGoalMultiBounce(Clone)",
            "HotZone(Clone)",
            "WallOut1",
            "WallOut2",
            "WallOut3",
            "WallOut4",
        };
        
        private Dictionary<string, int> targetMap = new Dictionary<string, int>();

        private int[]   resultIds       = new int[5];
        private float[] resultDistances = new float[5];
        private float[] result          = new float[10]; // idとdistanceを合わせたもの

        public float[] Result {
            get {
                // 前半にID, 後半にdistanceを入れて合わせたものとして返す.
                for(int i=0; i<resultIds.Length; ++i) {
                    result[i] = (float)resultIds[i];
                    result[i+resultIds.Length] = resultDistances[i];
                }
                return result;
            }
        }
        
        public LIDAR()
        {
            for(int i=0; i<targetNames.Length; ++i) {
                string targetName = targetNames[i];
                targetMap.Add(targetName, i);
            }

            ClearResult();
        }

        private string GetTargetName(Collider collided)
        {
            string name = collided.gameObject.name;
            if( name.StartsWith("Cube") || name == "RampFBX" ) {
                // LObjectとRampの場合はTopの1階層下にColliderが来るので、その上のObjectに
                // 欲しいオブジェクト名が入っているので例外的に処理する
                if( collided.gameObject.transform.parent != null ) {
                    name = collided.gameObject.transform.parent.gameObject.name;
                }
            }
            return name;
        }

        private Vector3 GetDirectionVec(Transform transform, float angleDegree) {
            float angleRad = Mathf.Deg2Rad * angleDegree;
            float c = Mathf.Cos(angleRad);
            float s = Mathf.Sin(angleRad);
            Vector3 v = new Vector3(s, 0.0f, c);
            Vector3 direction = transform.TransformDirection(v);
            return direction;
        }

        private void ProcessSub(int index, Transform transform) {
            // -20度〜20度まで10度きざみでスキャンする
            float angleDegree = -20.0f + 10.0f * index;
            
            Vector3 origin = transform.position;
            Vector3 direction = GetDirectionVec(transform, angleDegree);
            
            int targetId = -1;
            float targetDistance = -1.0f;
            
            RaycastHit hit;
            if( Physics.Raycast(origin, direction, out hit) ) {
                //Debug.DrawRay(transform.position, direction * hit.distance, Color.yellow);
                targetDistance = hit.distance;
                
                Collider collided = hit.collider;
                string targetName = GetTargetName(collided);
                if( targetMap.ContainsKey(targetName) ) {
                    targetId = targetMap[targetName];
                }
                if( targetId == -1 ) {
                    Debug.Log("target name=" + targetName + " id=" + targetId);
                }
            }
            // 見えない外壁が高いところまであるので、Raycastが失敗することは実際には無い

            resultIds[index] = targetId;
            resultDistances[index] = targetDistance;
        }

        private void ClearResult()
        {
            for(int i=0; i<resultIds.Length; ++i) {
                resultIds[i] = -1;
                resultDistances[i] = -1.0f;
            }
        }
        
        public void Process(Transform transform)
        {
            ClearResult();
            
            // -20度〜20度まで10度きざみでスキャンする
            for(int i=0; i<5; ++i ) {
                ProcessSub(i, transform);
            }
        }
    }
}
