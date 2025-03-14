import "./login.css"
import Image from "next/image"
import notes from "../../public/notes.png"

export default function Login() {
  return (
      <div>
        <header id="stripe">
          <h4>Compose yourself and take notes!</h4>
          <p className="fs-6 fw-lighter"><small>Login to access the music notes generator</small></p>
        </header>
        <div className="container text-center overflow-hidden pt-lg-5">
          <div className="row d-flex align-items-stretch h-100">
            <div className="col">
              <Image src={notes} alt="notes" width={900} height={70} id="picture"/>
            </div>
            <div className="col d-flex justify-content-center">
              <form className="w-50">
                <div className="row d-flex flex-row g-1 align-items-center">
                  <div className="col-4">
                    <label htmlFor="usernameBar" className="col-form-label">Username</label>
                  </div>
                  <div className="col">
                    <input type="username" id="usernameBar" className="form-control"/>
                  </div>
                </div>
                <div className="row d-flex flex-row g-1 align-items-center">
                  <div className="col-4">
                    <label htmlFor="passwordBar" className="col-form-label">Password</label>
                  </div>
                  <div className="col">
                    <input type="password" id="passwordBar" className="form-control"/>
                  </div>
                </div>
                <div className="col-5">
                  <button className="btn btn-primary" type="submit">Sign in</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        <div className="border-top border-dark" id="down">
          <div className="d-flex flex-column align-items-center justify-content-center mt-sm-5">
            <h5>New User?</h5>
            <p><small>Sign up and start using Taking Notes</small></p>
            <button type="button" className="btn btn-dark button">Sign Up</button>
          </div>
        </div>
      </div>
  );
}
